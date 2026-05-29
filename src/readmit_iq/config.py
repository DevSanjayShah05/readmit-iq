"""
Centralized configuration.

Reads environment variables (loaded from a .env file if present) and
exposes them as typed Python objects. The rest of the project should
import from here rather than calling os.getenv() ad-hoc, so we have a
single place to see every config knob the project recognizes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load .env from the project root, if present. Calling this at import time
# means every module that imports `config` automatically gets the .env values.
# python-dotenv is silent if .env is missing — useful in production where
# variables come from the orchestrator (Docker, Kubernetes) rather than a file.
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)


@dataclass(frozen=True)
class Settings:
    """All project configuration in one place."""
    app_env: str
    log_level: str
    data_root: Path
    database_url: str


def get_settings() -> Settings:
    """
    Build a Settings object from the current environment.

    Frozen dataclass + a single accessor: the rest of the project can read
    config values but can't mutate them, which avoids a class of bug where
    one module accidentally overwrites another's expectations.
    """
    settings = Settings(
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("APP_LOG_LEVEL", "INFO"),
        data_root=Path(os.getenv("DATA_ROOT", "./data")),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql://readmit:readmit@localhost:5432/readmit",
        ),
    )
    logger.debug(f"Loaded settings: env={settings.app_env} log={settings.log_level}")
    return settings
