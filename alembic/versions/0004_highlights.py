"""Highlight analyses and their detected candidates (M4).

Revision ID: 0004_highlights
Revises: 0003_transcripts
Create Date: 2026-11-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_highlights"
down_revision: str | None = "0003_transcripts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "highlights",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "video_id",
            sa.Text(),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("options_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
    )
    op.create_index("uq_highlights_video_id", "highlights", ["video_id"], unique=True)

    op.create_table(
        "highlight_candidates",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "highlight_id",
            sa.Text(),
            sa.ForeignKey("highlights.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("video_id", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
    )
    op.create_index(
        "ix_highlight_candidates_highlight_score",
        "highlight_candidates",
        ["highlight_id", "score"],
    )
    op.create_index("ix_highlight_candidates_video_id", "highlight_candidates", ["video_id"])


def downgrade() -> None:
    op.drop_index(
        "ix_highlight_candidates_video_id", table_name="highlight_candidates"
    )
    op.drop_index(
        "ix_highlight_candidates_highlight_score", table_name="highlight_candidates"
    )
    op.drop_table("highlight_candidates")
    op.drop_index("uq_highlights_video_id", table_name="highlights")
    op.drop_table("highlights")
