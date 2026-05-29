"""
Tests for readmit_iq.config
"""

from __future__ import annotations

import os

import pytest

from readmit_iq.config import Settings, get_settings


def test_settings_has_expected_fields() -> None:
    """Settings should expose app_env, log_level, data_root as attributes."""
    s = get_settings()
    assert hasattr(s, "app_env")
    assert hasattr(s, "log_level")
    assert hasattr(s, "data_root")


def test_settings_is_frozen() -> None:
    """Frozen dataclass: mutating a Settings instance should fail."""
    s = get_settings()
    with pytest.raises(Exception):  # FrozenInstanceError, subclass of AttributeError
        s.app_env = "production"  # type: ignore[misc]


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting an environment variable should be reflected in Settings."""
    monkeypatch.setenv("APP_ENV", "testing")
    s = get_settings()
    assert s.app_env == "testing"


def test_defaults_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If an env var is unset, the default should be used."""
    monkeypatch.delenv("APP_ENV", raising=False)
    s = get_settings()
    assert s.app_env == "development"
