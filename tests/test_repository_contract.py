"""Repository-level completeness contracts for causality-audit."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRENT_VERSION = "0.2.5"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_required_project_files_exist() -> None:
    for relative in [
        "README.md",
        "README.zh-CN.md",
        "CHANGELOG.md",
        "LICENSE",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
    ]:
        assert (ROOT / relative).is_file(), relative


def test_readmes_link_license_and_release_history() -> None:
    for relative in ["README.md", "README.zh-CN.md"]:
        text = _read(relative)
        assert "LICENSE" in text
        assert "CHANGELOG.md" in text


def test_changelog_documents_current_release() -> None:
    changelog = _read("CHANGELOG.md")
    assert f"## v{CURRENT_VERSION}" in changelog
    assert "Package resource links" in changelog


def test_version_is_consistent_across_package_metadata() -> None:
    pyproject = _read("pyproject.toml")
    package_init = _read("src/causality_audit/__init__.py")
    assert f'version = "{CURRENT_VERSION}"' in pyproject
    assert f'__version__ = "{CURRENT_VERSION}"' in package_init


def test_ci_runs_tests_builds_artifacts_and_smokes_console_script() -> None:
    ci = _read(".github/workflows/ci.yml")
    assert "python -m pytest" in ci
    assert "python -m build" in ci
    assert "actions/upload-artifact" in ci
    assert "causality-audit --version" in ci
    assert "causality-audit --demo buggy --json" in ci
    assert "SHA256SUMS.txt" in ci


def test_ci_runs_on_release_tags() -> None:
    ci = _read(".github/workflows/ci.yml")
    assert "tags:" in ci
    assert "v*" in ci


def test_package_metadata_links_project_resources() -> None:
    pyproject = _read("pyproject.toml")
    for label in ["Homepage", "Issues", "Changelog"]:
        assert f"{label} = " in pyproject


def test_codeql_workflow_is_enabled_for_python() -> None:
    codeql = _read(".github/workflows/codeql.yml")
    assert "github/codeql-action/init" in codeql
    assert "languages: python" in codeql
    assert "security-events: write" in codeql
