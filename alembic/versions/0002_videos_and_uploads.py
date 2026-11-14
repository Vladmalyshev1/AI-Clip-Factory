"""Videos and resumable uploads.

Revision ID: 0002_videos_and_uploads
Revises: 0001_initial_schema
Create Date: 2026-11-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_videos_and_uploads"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "videos",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False, server_default="upload"),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("video_codec", sa.Text(), nullable=True),
        sa.Column("audio_codec", sa.Text(), nullable=True),
        sa.Column("has_audio", sa.Integer(), nullable=True),
        sa.Column("container", sa.Text(), nullable=True),
        sa.Column("bitrate", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
    )
    op.create_index("ix_videos_status_created", "videos", ["status", "created_at"])
    op.create_index("ix_videos_sha256", "videos", ["sha256"])

    op.create_table(
        "uploads",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("total_bytes", sa.Integer(), nullable=False),
        sa.Column("received_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="uploading"),
        sa.Column("temp_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=True),
        sa.Column("video_id", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.Integer(), nullable=False),
    )
    op.create_index("ix_uploads_status_expires", "uploads", ["status", "expires_at"])


def downgrade() -> None:
    op.drop_index("ix_uploads_status_expires", table_name="uploads")
    op.drop_table("uploads")
    op.drop_index("ix_videos_sha256", table_name="videos")
    op.drop_index("ix_videos_status_created", table_name="videos")
    op.drop_table("videos")
