"""Clip editor drafts and export renders (M7).

Revision ID: 0006_clip_editor
Revises: 0005_clips
Create Date: 2026-11-17

An export is a clip: it reuses the ``clips`` table with ``kind='export'`` and a
``parent_clip_id`` pointing at the row it was cut from, so the queue, the render
stage and the file endpoints keep working unchanged. ``edit_json`` stores the
editor draft of one clip separately from the clip's own render.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_clip_editor"
down_revision: str | None = "0005_clips"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLite cannot ALTER TABLE ADD CONSTRAINT, so the table is rebuilt in batch
    # mode. The rebuild drops indexes and unnamed CHECK constraints; every one of
    # them is recreated below (named, so future batches can reflect them).
    with op.batch_alter_table("clips", recreate="always") as batch:
        batch.add_column(sa.Column("kind", sa.Text(), nullable=False, server_default="auto"))
        batch.add_column(sa.Column("parent_clip_id", sa.Text(), nullable=True))
        batch.add_column(sa.Column("edit_json", sa.Text(), nullable=True))
        batch.create_foreign_key(
            "fk_clips_parent_clip_id_clips",
            "clips",
            ["parent_clip_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.create_check_constraint(
            "ck_clips_start_non_negative", "start_ms >= 0"
        )
        batch.create_check_constraint("ck_clips_bounds_ordered", "end_ms > start_ms")
    # Only the two indexes that batch mode dropped are recreated here; the
    # M7 indexes are new and created in the downgrade branch instead.
    op.create_index("ix_clips_parent_clip_id", "clips", ["parent_clip_id"])
    op.create_index("ix_clips_kind_created", "clips", ["kind", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_clips_kind_created", table_name="clips")
    op.drop_index("ix_clips_parent_clip_id", table_name="clips")
    with op.batch_alter_table("clips", recreate="always") as batch:
        batch.drop_column("edit_json")
        batch.drop_column("parent_clip_id")
        batch.drop_column("kind")
        batch.create_check_constraint("ck_clips_start_non_negative", "start_ms >= 0")
        batch.create_check_constraint("ck_clips_bounds_ordered", "end_ms > start_ms")
