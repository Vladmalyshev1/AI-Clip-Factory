"""Application settings loaded from the environment (``ACF_`` prefix) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DB_FILENAME = "app.db"
DEFAULT_MAX_UPLOAD_BYTES = 8 * 1024**3
DEFAULT_UPLOAD_CHUNK_SIZE = 16 * 1024**2


class Settings(BaseSettings):
    """Runtime configuration for the API process and the worker process."""

    model_config = SettingsConfigDict(
        env_prefix="ACF_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    data_dir: Path = Path("var")
    database_url: str | None = None
    ffmpeg_path: str | None = None
    log_level: str = "INFO"
    log_dir: Path | None = None
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    worker_poll_seconds: float = 1.0
    worker_heartbeat_seconds: float = 5.0
    job_stale_seconds: float = 60.0
    job_max_attempts: int = 3
    min_free_disk_gb: float = 20.0
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 1.5
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    upload_chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE
    upload_ttl_hours: float = 24.0
    upload_cleanup_seconds: float = 300.0
    #: Directories ``POST /videos/import`` may read absolute paths from.
    #: An empty list denies every server-side import (uploads still work).
    import_roots: list[Path] = Field(default_factory=lambda: [Path.home()])

    # Speech recognition (M3). The model is resolved from the local Hugging Face
    # cache by default and is never downloaded implicitly: a missing model is
    # reported as a stage error with an explicit fetch instruction.
    asr_model: str = Field(default="Systran/faster-whisper-small", min_length=1)
    asr_device: str = Field(default="cpu", pattern="^(cpu|cuda|auto)$")
    asr_compute_type: str = Field(
        default="int8",
        pattern="^(default|auto|int8|int8_float32|int8_float16|int16|float16|float32|bfloat16)$",
    )
    asr_language: str = Field(default="en", min_length=2, max_length=8)
    asr_chunk_seconds: int = Field(default=300, ge=30, le=1800)
    asr_cpu_threads: int = Field(default=6, ge=0, le=64)
    asr_beam_size: int = Field(default=5, ge=1, le=10)
    asr_vad_filter: bool = True
    asr_local_files_only: bool = True
    asr_chunk_max_attempts: int = Field(default=3, ge=1, le=10)
    asr_condition_on_previous_text: bool = True

    # Highlight detection (M4). The detector consumes a completed transcript
    # only; every knob below shapes candidate boundaries and scoring.
    highlight_padding_ms: int = Field(default=2000, ge=0)
    highlight_merge_gap_ms: int = Field(default=10_000, ge=0)
    highlight_min_duration_ms: int = Field(default=15_000, ge=0)
    highlight_max_duration_ms: int = Field(default=120_000, ge=1000)
    highlight_context_window_ms: int = Field(default=30_000, ge=1000)
    highlight_min_score: float = Field(default=0.35, ge=0.0, le=1.0)
    highlight_max_candidates: int = Field(default=50, ge=1, le=500)

    @model_validator(mode="after")
    def _default_database_url(self) -> Settings:
        """Derive a SQLite URL under ``data_dir`` when none was configured.

        ``data_dir`` is resolved to an absolute path so the API and the worker
        agree on media/work/log locations even when started from different
        working directories (and the SQLite URL is always absolute).
        """
        self.data_dir = self.data_dir.expanduser().resolve()
        if not self.database_url:
            db_path = (self.data_dir / DEFAULT_DB_FILENAME).resolve()
            self.database_url = f"sqlite+pysqlite:///{db_path.as_posix()}"
        return self

    @property
    def media_root(self) -> Path:
        """Directory for source and rendered media."""
        return self.data_dir / "media"

    @property
    def incoming_dir(self) -> Path:
        """Directory holding partially received uploads (safe to purge)."""
        return self.media_root / "incoming"

    @property
    def originals_dir(self) -> Path:
        """Directory holding immutable source recordings."""
        return self.media_root / "originals"

    @property
    def renders_dir(self) -> Path:
        """Directory holding rendered vertical clips."""
        return self.media_root / "renders"

    @property
    def thumbs_dir(self) -> Path:
        """Directory holding generated thumbnails and previews."""
        return self.media_root / "thumbs"

    @property
    def work_dir(self) -> Path:
        """Directory for per-job scratch space."""
        return self.data_dir / "work"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files (``log_dir`` override wins)."""
        if self.log_dir is not None:
            return self.log_dir
        return self.data_dir / "logs"

    def media_dirs(self) -> tuple[Path, ...]:
        """Return every media subdirectory in creation order."""
        return (
            self.media_root,
            self.incoming_dir,
            self.originals_dir,
            self.renders_dir,
            self.thumbs_dir,
        )

    def ensure_dirs(self) -> None:
        """Create every directory the application writes to."""
        for path in (*self.media_dirs(), self.data_dir, self.work_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
