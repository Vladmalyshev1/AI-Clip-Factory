"""Live integration: run a real server, exercise the M7 flow over HTTP.

Started as a script with the repo on sys.path; uses a temp data dir and a real
FFmpeg-generated source recording.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = Path(os.environ["TEMP"]) / "acf-live-m7"
shutil.rmtree(DATA, ignore_errors=True)
DATA.mkdir(parents=True)

from app.core.config import Settings  # noqa: E402

settings = Settings(data_dir=DATA)
settings.ensure_dirs()

# --- generate a real source recording -------------------------------------
from app.media.binaries import require_ffmpeg  # noqa: E402

source = DATA.parent / "acf-live-source.mp4"
subprocess.run(
    [
        str(require_ffmpeg(settings)), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(source),
    ],
    check=True,
)

from app.core.clock import utc_now_ms  # noqa: E402
from app.core.db import init_schema, make_engine, make_session_factory  # noqa: E402
from app.core.status import SourceKind, VideoStatus  # noqa: E402
from app.models.job import new_id  # noqa: E402
from app.models.video import Video  # noqa: E402

engine = make_engine(settings)
init_schema(engine)
factory = make_session_factory(engine)
video_id = new_id()
now = utc_now_ms()
with factory() as session:
    session.add(
        Video(
            id=video_id,
            title="Live VOD",
            original_filename="vod.mp4",
            source_path=str(source),
            source_kind=SourceKind.IMPORT_IN_PLACE.value,
            size_bytes=source.stat().st_size,
            duration_ms=3_000,
            has_audio=1,
            status=VideoStatus.READY.value,
            created_at=now,
            updated_at=now,
        )
    )
    session.commit()

os.environ["ACF_DATA_DIR"] = str(DATA)

# --- start the API server --------------------------------------------------
server = subprocess.Popen(
    [
        str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn",
        "app.main:app", "--port", "8901", "--log-level", "warning",
    ],
    cwd=ROOT,
    env={**os.environ},
)
try:
    import urllib.error
    import urllib.request

    base = "http://127.0.0.1:8901/api/v1"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    else:
        raise RuntimeError("server did not start")

    # Range request against the source endpoint.
    request = urllib.request.Request(f"{base}/videos/{video_id}/source")
    request.add_header("Range", "bytes=0-99")
    with urllib.request.urlopen(request) as response:
        assert response.status == 206, response.status
        body = response.read()
        assert len(body) == 100, len(body)
        assert response.headers["Content-Range"].startswith("bytes 0-99/")
    print("range 206: ok")

    # Editor + export round trip on a seeded clip.
    from app.models.clip import Clip

    clip_id = new_id()
    with factory() as session:
        session.add(
            Clip(
                id=clip_id,
                video_id=video_id,
                kind="auto",
                status="completed",
                title="live clip",
                start_ms=0,
                end_ms=2_000,
                duration_ms=2_000,
                output_format="vertical",
                crop_mode="pad",
                encoding_mode="accurate",
                width=1080,
                height=1920,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()

    request = urllib.request.Request(
        f"{base}/clips/{clip_id}/editor",
        data=json.dumps(
            {
                "start_ms": 200,
                "end_ms": 1_800,
                "subtitles": {
                    "enabled": True,
                    "style": "gaming",
                    "font_size": 60,
                    "cues": [
                        {"start_ms": 0, "end_ms": 800, "text": "live cue один"},
                        {"start_ms": 800, "end_ms": 1_600, "text": "second cue"},
                    ],
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 200
        saved = json.loads(response.read())
    assert saved["subtitles"]["enabled"] is True
    print("editor PUT: ok")

    request = urllib.request.Request(
        f"{base}/clips/{clip_id}/exports", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 201
        export = json.loads(response.read())["items"][0]
    export_id = export["id"]
    assert export["start_ms"] == 200 and export["end_ms"] == 1_800
    print("export created:", export_id)

    # Render it synchronously through the real stage.
    from app.pipeline.stages.base import ProgressReporter
    from app.pipeline.stages.clips import RenderClipStage

    class Reporter(ProgressReporter):
        def update(self, progress, message=None):
            pass

        def check_cancelled(self):
            return None

    job_id = export["job_id"]
    ctx = type(
        "Ctx",
        (),
        {
            "job_id": job_id,
            "video_id": video_id,
            "attempt": 1,
            "work_dir": settings.work_dir,
            "params": {"clip_id": export_id},
            "checkpoint": None,
            "settings": settings,
            "session_factory": factory,
        },
    )()
    RenderClipStage().run(ctx, Reporter())

    with urllib.request.urlopen(f"{base}/exports/{export_id}") as response:
        final = json.loads(response.read())
    assert final["status"] == "completed", final
    path = settings.clips_dir / video_id / f"{export_id}.mp4"
    assert path.is_file(), path
    print("export rendered:", final["width"], "x", final["height"], path.stat().st_size, "bytes")

    with urllib.request.urlopen(f"{base}/exports/{export_id}/file") as response:
        assert response.status == 200
        assert len(response.read()) > 1000
    print("export download: ok")

    # Library endpoint shows both auto clip and export.
    with urllib.request.urlopen(f"{base}/clips?kind=export") as response:
        listing = json.loads(response.read())
    assert listing["total"] == 1
    print("library filter: ok")

    print("LIVE M7 FLOW: ALL OK")
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
