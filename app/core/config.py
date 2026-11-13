"""Application settings loaded from the environment (``ACF_`` prefix) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DB_FILENAME = "app.db"


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
    def work_dir(self) -> Path:
        """Directory for per-job scratch space."""
        return self.data_dir / "work"

    @property
    def logs_dir(self) -> Path:
        """Directory for log files (``log_dir`` override wins)."""
        if self.log_dir is not None:
            return self.log_dir
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        """Create every directory the application writes to."""
        for path in (self.data_dir, self.media_root, self.work_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
