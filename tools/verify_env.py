"""Environment readiness checks for AI Clip Factory.

Deliberately dependency-free (stdlib only) so it can run before the project
virtualenv exists and inside a broken environment.

Usage:
    py -3.12 tools/verify_env.py
    py -3.12 tools/verify_env.py --json
    py -3.12 tools/verify_env.py --strict   # WARN becomes failure

Exit codes:
    0  all required checks passed
    1  one or more checks failed
    10 wrong/missing Python interpreter
    11 ffmpeg not found
    12 ffmpeg found but built without libass / required filters
    13 ffprobe not found
    14 ffmpeg lacks a required encoder
    15 GPU / NVENC not usable (warning by default)
    16 not enough free disk space
    17 long paths disabled and the workspace path is long
    18 Ollama not reachable (warning by default)
    19 Node.js missing or too old
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_PYTHON = (3, 12)
MIN_FFMPEG = (6, 1)
MIN_NODE_MAJOR = 20

MIN_FREE_GB = 20
COMFORTABLE_FREE_GB = 60
LONG_PATH_THRESHOLD = 200

REQUIRED_FILTERS = ("ass", "subtitles", "gblur", "crop", "overlay", "loudnorm")
REQUIRED_ENCODERS = ("libx264", "aac")

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    code: int = 0
    hint: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    root: Path = field(default_factory=Path.cwd)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]


def _creation_flags() -> int:
    if sys.platform != "win32":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    """Run a command and return (exit_code, combined_output). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_creation_flags(),
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "executable not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except OSError as exc:  # pragma: no cover - platform specific
        return 126, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- paths


def resolve_ffmpeg(root: Path) -> Path | None:
    candidates: list[Path] = []
    env_path = os.environ.get("ACF_FFMPEG")
    if env_path:
        candidates.append(Path(env_path))
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidates.append(root / "tools" / "ffmpeg" / "bin" / exe)
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_ffprobe(ffmpeg: Path | None, root: Path) -> Path | None:
    exe = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    candidates: list[Path] = []
    if ffmpeg is not None:
        candidates.append(ffmpeg.with_name(exe))
    candidates.append(root / "tools" / "ffmpeg" / "bin" / exe)
    found = shutil.which("ffprobe")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- checks


def check_python() -> Check:
    version = sys.version_info
    if version[:2] != REQUIRED_PYTHON:
        return Check(
            "python",
            FAIL,
            f"running {version.major}.{version.minor}.{version.micro}",
            code=10,
            hint=(
                "Project requires Python 3.12. Use: py -3.12 -m venv .venv "
                "and call .venv\\Scripts\\python.exe explicitly."
            ),
        )
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        return Check(
            "python",
            WARN,
            f"{version.major}.{version.minor}.{version.micro} (not in a virtualenv)",
            hint="Run inside .venv to avoid polluting the global interpreter.",
        )
    return Check("python", OK, f"{version.major}.{version.minor}.{version.micro} (venv)")


def check_ffmpeg(ffmpeg: Path | None) -> tuple[Check, tuple[int, int] | None]:
    if ffmpeg is None:
        return (
            Check(
                "ffmpeg",
                FAIL,
                "not found",
                code=11,
                hint="Run tools\\setup.ps1 or set ACF_FFMPEG to ffmpeg.exe",
            ),
            None,
        )
    code, out = run([str(ffmpeg), "-hide_banner", "-version"])
    if code != 0:
        return Check("ffmpeg", FAIL, f"cannot execute: {out.strip()[:120]}", code=11), None
    match = re.search(r"ffmpeg version (?:n)?(\d+)\.(\d+)", out)
    if not match:
        return Check("ffmpeg", FAIL, "unrecognised version string", code=11), None
    version = (int(match.group(1)), int(match.group(2)))
    if version < MIN_FFMPEG:
        return (
            Check(
                "ffmpeg",
                FAIL,
                f"{version[0]}.{version[1]} < required {MIN_FFMPEG[0]}.{MIN_FFMPEG[1]}",
                code=11,
            ),
            version,
        )
    return Check("ffmpeg", OK, f"{version[0]}.{version[1]} at {ffmpeg}"), version


def check_libass(ffmpeg: Path | None) -> Check:
    if ffmpeg is None:
        return Check("ffmpeg:libass", SKIP, "ffmpeg missing")
    _, buildconf = run([str(ffmpeg), "-hide_banner", "-buildconf"])
    _, filters = run([str(ffmpeg), "-hide_banner", "-filters"])
    has_build_flag = "--enable-libass" in buildconf
    missing = [name for name in REQUIRED_FILTERS if not re.search(rf"(?m)^\s*\S+\s+{re.escape(name)}\s", filters)]
    if not has_build_flag or missing:
        return Check(
            "ffmpeg:libass",
            FAIL,
            f"buildconf_libass={has_build_flag}, missing filters={missing or 'none'}",
            code=12,
            hint="Use a GPL build with libass (BtbN win64-gpl or Gyan full).",
        )
    return Check("ffmpeg:libass", OK, "libass + required filters present")


def check_encoders(ffmpeg: Path | None) -> Check:
    if ffmpeg is None:
        return Check("ffmpeg:encoders", SKIP, "ffmpeg missing")
    _, out = run([str(ffmpeg), "-hide_banner", "-encoders"])
    missing = [name for name in REQUIRED_ENCODERS if name not in out]
    if missing:
        return Check("ffmpeg:encoders", FAIL, f"missing {missing}", code=14)
    has_nvenc = "h264_nvenc" in out
    return Check(
        "ffmpeg:encoders",
        OK,
        f"libx264+aac present, nvenc={'yes' if has_nvenc else 'no'}",
    )


def check_ffprobe(ffprobe: Path | None) -> Check:
    if ffprobe is None:
        return Check("ffprobe", FAIL, "not found", code=13, hint="Must sit next to ffmpeg")
    code, out = run([str(ffprobe), "-hide_banner", "-version"])
    if code != 0:
        return Check("ffprobe", FAIL, f"cannot execute: {out.strip()[:120]}", code=13)
    return Check("ffprobe", OK, out.splitlines()[0][:80] if out else "ok")


def check_nvenc_trial(ffmpeg: Path | None) -> Check:
    if ffmpeg is None:
        return Check("gpu:nvenc", SKIP, "ffmpeg missing")
    nul = "NUL" if sys.platform == "win32" else "/dev/null"
    code, out = run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30",
            "-t",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            nul,
        ],
        timeout=90,
    )
    if code != 0:
        return Check(
            "gpu:nvenc",
            WARN,
            f"NVENC encode failed: {out.strip()[:120] or 'unknown error'}",
            code=15,
            hint="CPU encoding (libx264) is the MVP default; NVENC is optional.",
        )
    return Check("gpu:nvenc", OK, "h264_nvenc trial encode succeeded")


def check_gpu() -> Check:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return Check("gpu", WARN, "nvidia-smi not found", code=15)
    code, out = run(
        [
            smi,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if code != 0:
        return Check("gpu", WARN, f"nvidia-smi failed: {out.strip()[:120]}", code=15)
    return Check("gpu", OK, out.strip().replace("\n", "; "))


def check_disk(root: Path) -> Check:
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    if free_gb < MIN_FREE_GB:
        return Check("disk", FAIL, f"{free_gb:.1f} GB free", code=16, hint=f"Need at least {MIN_FREE_GB} GB")
    if free_gb < COMFORTABLE_FREE_GB:
        return Check("disk", WARN, f"{free_gb:.1f} GB free")
    return Check("disk", OK, f"{free_gb:.1f} GB free")


def check_long_paths(root: Path) -> Check:
    if sys.platform != "win32":
        return Check("long_paths", SKIP, "not Windows")
    enabled = 0
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            enabled = int(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError as exc:
        return Check("long_paths", WARN, f"cannot read registry: {exc}")
    depth = len(str(root))
    if enabled:
        return Check("long_paths", OK, f"enabled (workspace path {depth} chars)")
    if depth > LONG_PATH_THRESHOLD:
        return Check(
            "long_paths",
            FAIL,
            f"disabled and workspace path is {depth} chars",
            code=17,
            hint="Move the project closer to the drive root or enable long paths.",
        )
    return Check("long_paths", WARN, f"disabled (workspace path {depth} chars)")


def check_ollama() -> Check:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
            pass
    except OSError:
        return Check(
            "ollama",
            WARN,
            "not reachable on 127.0.0.1:11434",
            code=18,
            hint="Optional. Rule-based metadata fallback must work without it.",
        )
    return Check("ollama", OK, "reachable on 127.0.0.1:11434")


def check_node() -> Check:
    node = shutil.which("node")
    if not node:
        return Check("node", WARN, "not found", code=19, hint="Required only for the frontend.")
    code, out = run([node, "--version"])
    if code != 0:
        return Check("node", WARN, f"cannot execute: {out.strip()[:80]}", code=19)
    match = re.search(r"v(\d+)", out)
    major = int(match.group(1)) if match else 0
    if major < MIN_NODE_MAJOR:
        return Check("node", WARN, f"{out.strip()} < {MIN_NODE_MAJOR}", code=19)
    return Check("node", OK, out.strip())


def check_locked_requirements(root: Path) -> Check:
    lock = root / "requirements.lock.txt"
    if not lock.is_file():
        return Check(
            "requirements.lock",
            WARN,
            "missing",
            hint="Run tools\\setup.ps1 to freeze verified versions.",
        )
    return Check("requirements.lock", OK, f"{lock.stat().st_size} bytes")


# --------------------------------------------------------------------------- main


def run_checks(root: Path) -> Report:
    report = Report(root=root)
    ffmpeg = resolve_ffmpeg(root)
    ffprobe = resolve_ffprobe(ffmpeg, root)

    report.add(check_python())
    ffmpeg_check, _ = check_ffmpeg(ffmpeg)
    report.add(ffmpeg_check)
    report.add(check_libass(ffmpeg))
    report.add(check_encoders(ffmpeg))
    report.add(check_ffprobe(ffprobe))
    report.add(check_gpu())
    report.add(check_nvenc_trial(ffmpeg))
    report.add(check_disk(root))
    report.add(check_long_paths(root))
    report.add(check_ollama())
    report.add(check_node())
    report.add(check_locked_requirements(root))
    return report


def print_table(report: Report) -> None:
    name_width = max(len(c.name) for c in report.checks) + 2
    status_width = 6
    print()
    print(f"{'CHECK'.ljust(name_width)}{'STATUS'.ljust(status_width)}DETAIL")
    print("-" * (name_width + status_width + 40))
    for check in report.checks:
        print(f"{check.name.ljust(name_width)}{check.status.ljust(status_width)}{check.detail}")
    print("-" * (name_width + status_width + 40))
    failed = report.failed
    warned = report.warned
    print(f"{len(report.checks)} checks: {len(failed)} failed, {len(warned)} warnings")
    for check in failed:
        if check.hint:
            print(f"  [{check.code}] {check.name}: {check.hint}")
    for check in warned:
        if check.hint:
            print(f"  [{check.code}] {check.name}: {check.hint}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check AI Clip Factory environment readiness.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)

    report = run_checks(Path(args.root).resolve())
    exit_code = 0
    if report.failed:
        exit_code = report.failed[0].code or 1
    elif args.strict and report.warned:
        exit_code = report.warned[0].code or 1

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(report.root),
                    "ok": exit_code == 0,
                    "checks": [c.__dict__ for c in report.checks],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print_table(report)
        if exit_code == 0:
            print("Environment OK.")
        else:
            print(f"Environment NOT ready (exit {exit_code}).")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
