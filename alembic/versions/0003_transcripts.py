"""Transcripts, their resumable chunks and recognised segments.

Revision ID: 0003_transcripts
Revises: 0002_videos_and_uploads
Create Date: 2026-11-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_transcripts"
down_revision: str | None = "0002_videos_and_uploads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "transcripts",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "video_id",
            sa.Text(),
            sa.ForeignKey("videos.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("language", sa.Text(), nullable=False, server_default="en"),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("chunk_seconds", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("options_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
    )
    op.create_index("uq_transcripts_video_id", "transcripts", ["video_id"], unique=True)

    op.create_table(
        "transcript_chunks",
        sa.Column(
            "transcript_id",
            sa.Text(),
            sa.ForeignKey("transcripts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("chunk_index", sa.Integer(), primary_key=True),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("segment_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processed_at", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
    )

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "transcript_id",
            sa.Text(),
            sa.ForeignKey("transcripts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("avg_logprob", sa.Float(), nullable=True),
        sa.Column("no_speech_prob", sa.Float(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
    )
    op.create_index(
        "ix_transcript_segments_transcript_start",
        "transcript_segments",
        ["transcript_id", "start_ms"],
    )
    op.create_index(
        "ix_transcript_segments_transcript_chunk",
        "transcript_segments",
        ["transcript_id", "chunk_index"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_transcript_segments_transcript_chunk", table_name="transcript_segments"
    )
    op.drop_index("ix_transcript_segments_transcript_start", table_name="transcript_segments")
    op.drop_table("transcript_segments")
    op.drop_table("transcript_chunks")
    op.drop_index("uq_transcripts_video_id", table_name="transcripts")
    op.drop_table("transcripts")
