"""
Alembic environment.

We override the default URL handling to read DATABASE_URL from our app
settings (which themselves read from .env). This keeps secrets out of
the committed alembic.ini and uses one source of truth for the URL.
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from readmit_iq.config import get_settings

# Alembic Config object — gives access to the alembic.ini values.
config = context.config

# Override the URL from our app settings (which loaded .env).
settings = get_settings()
# SQLAlchemy needs the dialect+driver scheme to pick psycopg3 instead of the
# (uninstalled) psycopg2. We store the URL in standard postgresql:// form for
# everyone else, and rewrite it here at the SQLAlchemy boundary.
sqlalchemy_url = settings.database_url.replace(
    "postgresql://", "postgresql+psycopg://", 1
)
config.set_main_option("sqlalchemy.url", sqlalchemy_url)
# Set up Python's logging from alembic.ini's [loggers] sections.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# We're not using SQLAlchemy models for autogeneration yet, so target_metadata
# stays None. Migrations will be written by hand — clearer for learning.
target_metadata = None


def run_migrations_offline() -> None:
    """Generate SQL scripts without connecting to a database."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations live."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
