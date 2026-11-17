"""Generated clips and their render metadata (M6).

Revision ID: 0005_clips
Revises: 0004_highlights
Create Date: 2026-11-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_clips"
down_revision: str | None = "0004_highlights"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "clips",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "video_id",
            sa.Text(),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "highlight_id",
            sa.Text(),
            sa.ForeignKey("highlights.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "highlight_candidate_id",
            sa.Text(),
            sa.ForeignKey("highlight_candidates.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Float(), nullable=True),
        sa.Column("aspect_ratio", sa.Text(), nullable=True),
        sa.Column("encoding_mode", sa.Text(), nullable=True),
        sa.Column("output_format", sa.Text(), nullable=True),
        sa.Column("crop_mode", sa.Text(), nullable=True),
        sa.Column("options_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("job_id", sa.Text(), nullable=True),
        sa.Column("started_at", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.CheckConstraint("start_ms >= 0", name="ck_clips_start_non_negative"),
        sa.CheckConstraint("end_ms > start_ms", name="ck_clips_bounds_ordered"),
    )
    op.create_index("ix_clips_video_id", "clips", ["video_id"])
    op.create_index("ix_clips_status_created", "clips", ["status", "created_at"])
    op.create_index("uq_clips_candidate_id", "clips", ["highlight_candidate_id"], unique=True)


def downgrade() -> None:
    op.drop_index("uq_clips_candidate_id", table_name="clips")
    op.drop_index("ix_clips_status_created", table_name="clips")
    op.drop_index("ix_clips_video_id", table_name="clips")
    op.drop_table("clips")
