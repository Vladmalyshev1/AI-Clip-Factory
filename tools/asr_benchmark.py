"""Benchmark the local speech-recognition pipeline on a real recording.

The tool mirrors what the worker does (extract 16 kHz mono, slice fixed-length
chunks, decode each chunk with faster-whisper) and reports the realtime factor
(RTF = wall seconds / audio seconds) per chunk, so an estimate for a 2-5 hour VOD
rests on measurements from this machine instead of a guess.

Usage:
    .\\.venv\\Scripts\\python.exe tools/asr_benchmark.py path\\to\\recording.mp4
    .\\.venv\\Scripts\\python.exe tools/asr_benchmark.py recording.mp4 --limit-seconds 600
    .\\.venv\\Scripts\\python.exe tools/asr_benchmark.py recording.mp4 --json

Exit codes:
    0  benchmark finished
    1  the source is unreadable, FFmpeg is missing or the model is not available
    2  invalid arguments

The source file is never modified: audio is extracted into a temporary directory
that is removed afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(ROOT))

from app.ai.asr import WhisperAsrEngine  # noqa: E402
from app.ai.model_manager import AsrModelManager  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.errors import AppError  # noqa: E402
from app.media import audio as audio_io  # noqa: E402
from app.media import audio_chunks  # noqa: E402
from app.media.runner import FFmpegRunner, RenderError  # noqa: E402

DEFAULT_LIMIT_SECONDS = 600.0
MIN_LIMIT_SECONDS = 10.0


@dataclass
class ChunkMeasurement:
    """What one chunk cost."""

    index: int
    start_ms: int
    end_ms: int
    audio_s: float
    wall_s: float
    decode_s: float
    rtf: float
    segments: int
    chars: int


@dataclass
class BenchmarkReport:
    """Aggregated benchmark result, printable as a table or as JSON."""

    source: str
    source_duration_ms: int
    analysed_ms: int
    chunk_seconds: int
    model: str
    device: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    vad_filter: bool
    model_load_s: float
    prepare_s: float
    total_audio_s: float
    total_wall_s: float
    rtf: float
    segments: int
    chars: int
    peak_rss_mb: float | None
    chunks: list[ChunkMeasurement] = field(default_factory=list)


def _is_target_wav(path: Path) -> bool:
    """Return whether ``path`` is already a mono 16 kHz WAV (no extraction)."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getframerate() == audio_io.TARGET_SAMPLE_RATE and (
                handle.getnchannels() == audio_io.TARGET_CHANNELS
            )
    except (OSError, EOFError, ValueError, wave.Error):
        return False


def _prepare_audio(settings: Settings, source: Path, workdir: Path) -> Path:
    """Return a mono 16 kHz WAV for ``source``, extracting one if needed."""
    if _is_target_wav(source):
        return source
    target = workdir / audio_io.AUDIO_FILENAME
    audio_io.extract_audio(FFmpegRunner.from_settings(settings), source, target)
    return target


def _current_rss_mb() -> float | None:
    """Return the current RSS in MiB when psutil is available."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is installed in the venv
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - defensive
        return None


def _build_engine(settings: Settings, manager: AsrModelManager) -> WhisperAsrEngine:
    """Build the same engine the worker uses, on top of ``manager``."""
    return WhisperAsrEngine(
        manager,
        beam_size=settings.asr_beam_size,
        vad_filter=settings.asr_vad_filter,
        condition_on_previous_text=settings.asr_condition_on_previous_text,
        language=settings.asr_language,
    )


def _run(
    settings: Settings,
    engine: WhisperAsrEngine,
    audio_path: Path,
    *,
    chunks_dir: Path,
    chunk_seconds: int,
    limit_seconds: float,
) -> BenchmarkReport:
    """Transcribe the first ``limit_seconds`` of ``audio_path`` and measure it."""
    duration_ms = audio_io.wav_duration_ms(audio_path)
    if duration_ms is None:
        raise ValueError(f"Cannot read the duration of {audio_path}")
    limit_ms = min(duration_ms, int(limit_seconds * 1000))

    runner = FFmpegRunner.from_settings(settings)
    plans = audio_chunks.plan_chunks(limit_ms, chunk_seconds)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    measurements: list[ChunkMeasurement] = []
    peak_rss = _current_rss_mb()
    started_all = time.monotonic()
    try:
        for plan in plans:
            target = audio_chunks.chunk_path(chunks_dir, plan.index)
            try:
                started = time.monotonic()
                audio_chunks.slice_chunk(runner, audio_path, target, plan)
                samples = audio_io.read_audio_16k(target)
                decode_started = time.monotonic()
                result = engine.transcribe(
                    samples,
                    offset_ms=plan.start_ms,
                    duration_ms=plan.duration_ms,
                )
                wall_s = time.monotonic() - started
            finally:
                audio_chunks.remove_file(target)

            audio_s = plan.duration_ms / 1000.0
            measurement = ChunkMeasurement(
                index=plan.index,
                start_ms=plan.start_ms,
                end_ms=plan.end_ms,
                audio_s=audio_s,
                wall_s=wall_s,
                decode_s=time.monotonic() - decode_started,
                rtf=wall_s / audio_s if audio_s > 0 else 0.0,
                segments=len(result.segments),
                chars=sum(len(segment.text) for segment in result.segments),
            )
            measurements.append(measurement)
            current_rss = _current_rss_mb()
            if current_rss is not None:
                peak_rss = max(peak_rss or 0.0, current_rss)
            print(
                f"chunk {plan.index:>3}  audio {audio_s:7.1f}s  "
                f"wall {wall_s:7.2f}s  decode {measurement.decode_s:7.2f}s  "
                f"RTF {measurement.rtf:5.3f}  segments {measurement.segments:>4}",
                flush=True,
            )
    finally:
        audio_chunks.cleanup_chunks(chunks_dir)
        shutil.rmtree(chunks_dir, ignore_errors=True)

    total_audio = sum(item.audio_s for item in measurements)
    total_wall = time.monotonic() - started_all
    return BenchmarkReport(
        source="",
        source_duration_ms=duration_ms,
        analysed_ms=limit_ms,
        chunk_seconds=chunk_seconds,
        model=engine.model_name,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        cpu_threads=settings.asr_cpu_threads,
        beam_size=settings.asr_beam_size,
        vad_filter=settings.asr_vad_filter,
        model_load_s=0.0,
        prepare_s=0.0,
        total_audio_s=total_audio,
        total_wall_s=total_wall,
        rtf=total_wall / total_audio if total_audio > 0 else 0.0,
        segments=sum(item.segments for item in measurements),
        chars=sum(item.chars for item in measurements),
        peak_rss_mb=peak_rss,
        chunks=measurements,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/asr_benchmark.py",
        description="Measure local speech recognition speed on a real recording.",
    )
    parser.add_argument("source", type=Path, help="audio or video file to transcribe")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=None,
        help="chunk length (default: ACF_ASR_CHUNK_SECONDS)",
    )
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=DEFAULT_LIMIT_SECONDS,
        help=f"analyse only the first N seconds (default {DEFAULT_LIMIT_SECONDS:g})",
    )
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")
    return parser


def _print_report(report: BenchmarkReport) -> None:
    """Print the human readable summary."""
    print()
    if report.rtf > 0:
        print(
            f"total       : audio {report.total_audio_s:.1f}s, "
            f"wall {report.total_wall_s:.2f}s, RTF {report.rtf:.3f} "
            f"({1 / report.rtf:.1f}x realtime)"
        )
        print(
            f"estimate    : {report.rtf * 60:.1f} min of CPU time per hour of recording"
        )
    else:
        print("total       : no audio was analysed")
    print(
        f"segments    : {report.segments} ({report.chars} chars), "
        f"model load {report.model_load_s:.2f}s, "
        f"peak RSS {report.peak_rss_mb} MB"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a report."""
    args = _build_parser().parse_args(argv)
    if args.limit_seconds < MIN_LIMIT_SECONDS:
        print(
            f"error: --limit-seconds must be at least {MIN_LIMIT_SECONDS:g}",
            file=sys.stderr,
        )
        return 2
    source = args.source.expanduser()
    if not source.is_file():
        print(f"error: {source} is not a file", file=sys.stderr)
        return 1

    settings = get_settings()
    chunk_seconds = args.chunk_seconds or settings.asr_chunk_seconds
    workdir = Path(tempfile.mkdtemp(prefix="acf-asr-bench-"))
    manager = AsrModelManager(
        model=settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        cpu_threads=settings.asr_cpu_threads,
        local_files_only=settings.asr_local_files_only,
    )
    prepare_started = time.monotonic()
    try:
        audio_path = _prepare_audio(settings, source, workdir)
        load_started = time.monotonic()
        manager.get_model()
        load_s = time.monotonic() - load_started
        prepare_s = time.monotonic() - prepare_started
        if not args.json:
            print(f"source      : {source}")
            print(f"audio       : {audio_path}")
            print(
                f"model       : {settings.asr_model} "
                f"({settings.asr_device}/{settings.asr_compute_type}, "
                f"{settings.asr_cpu_threads} threads, beam {settings.asr_beam_size}, "
                f"vad {settings.asr_vad_filter})"
            )
            print(f"model load  : {load_s:.2f}s (preparation {prepare_s:.2f}s)")
            print()
        report = _run(
            settings,
            _build_engine(settings, manager),
            audio_path,
            # Never write next to the user's recording: an existing
            # ``bench-chunks`` directory there must not be cleaned up.
            chunks_dir=workdir / "bench-chunks",
            chunk_seconds=chunk_seconds,
            limit_seconds=args.limit_seconds,
        )
    except AppError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    except (RenderError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report.source = str(source)
    report.model_load_s = round(load_s, 3)
    report.prepare_s = round(prepare_s, 3)
    if args.json:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        return 0
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
