"""Regression tests for current setuptools license metadata.

Setuptools 77+ deprecates the legacy ``license = {text = ...}`` table
and license classifiers. Keep the project metadata in SPDX form so local
builds and CI stay warning-clean as packaging tooling tightens.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def _pyproject_text() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def test_project_license_uses_spdx_string() -> None:
    text = _pyproject_text()
    assert 'license = "MIT"' in text
    assert "license = {" not in text


def test_license_file_is_declared() -> None:
    text = _pyproject_text()
    assert 'license-files = ["LICENSE"]' in text


def test_deprecated_license_classifier_is_absent() -> None:
    text = _pyproject_text()
    assert "License :: OSI Approved :: MIT License" not in text


def test_build_backend_floor_supports_spdx_license_metadata() -> None:
    text = _pyproject_text()
    assert '"setuptools>=77"' in text
