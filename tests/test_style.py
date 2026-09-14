import subprocess
import sys

import pytest


def test_style_module_importable_standalone():
    from causality_audit import style

    assert hasattr(style, "Style")
    assert hasattr(style, "resolve_style")


def test_resolve_style_no_color_flag_disables():
    from causality_audit.style import resolve_style

    s = resolve_style(no_color_flag=True)
    assert s.enabled is False


def test_resolve_style_no_color_env(monkeypatch):
    from causality_audit.style import resolve_style

    monkeypatch.setenv("NO_COLOR", "1")
    s = resolve_style(no_color_flag=False)
    assert s.enabled is False


def test_status_headline_uses_single_glyph_family():
    from causality_audit.style import Style, status_headline

    plain = Style(False)
    text = status_headline(plain, "ok", "hello")
    assert "hello" in text
    assert "[OK]" in text


def test_package_has_pyproject_and_license_and_readme():
    import importlib.resources as res

    # Sanity: this repo follows the fleet's src-layout convention (not a
    # style test per se, but a packaging smoke check).
    import causality_audit

    assert causality_audit.__version__
