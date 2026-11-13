"""Initial schema: jobs, job_stages, job_events.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-13
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("type", sa.Text(), nullable=False, server_default="pipeline"),
        sa.Column("video_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("current_stage", sa.Text(), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("params_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("worker_id", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("heartbeat_at", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.Integer(), nullable=False),
    )
    op.execute(
        "CREATE INDEX ix_jobs_status_priority_created "
        "ON jobs (status, priority DESC, created_at)"
    )
    op.create_index("ix_jobs_video_id", "jobs", ["video_id"])

    op.create_table(
        "job_stages",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "job_id",
            sa.Text(),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("checkpoint_json", sa.Text(), nullable=True),
        sa.Column("output_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.Integer(), nullable=True),
        sa.Column("finished_at", sa.Integer(), nullable=True),
        sa.UniqueConstraint("job_id", "name", name="uq_job_stages_job_id_name"),
    )

    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            sa.Text(),
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ts", sa.Integer(), nullable=False),
        sa.Column("level", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=True),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_job_events_job_id_id", "job_events", ["job_id", "id"])


def downgrade() -> None:
    op.drop_index("ix_job_events_job_id_id", table_name="job_events")
    op.drop_table("job_events")
    op.drop_table("job_stages")
    op.drop_index("ix_jobs_video_id", table_name="jobs")
    op.execute("DROP INDEX ix_jobs_status_priority_created")
    op.drop_table("jobs")
