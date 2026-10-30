"""Окружение Alembic. Миграции запускаются из кода при старте приложения."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from orbita.storage.orm import Base

url = context.config.get_main_option("sqlalchemy.url")
assert url is not None
engine = create_engine(url)

with engine.connect() as connection:
    # batch-режим нужен SQLite: ALTER TABLE там почти ничего не умеет.
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
engine.dispose()
