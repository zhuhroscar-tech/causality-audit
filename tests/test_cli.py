"""CLI tests: --demo clean/buggy, --json, --check-clean exit codes,
--no-color, --epsilon-sweep."""
from __future__ import annotations

import json

import pytest

from causality_audit.cli import main


def test_help_exits_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "causality-audit" in out or "usage" in out.lower()


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_demo_clean_text_output(capsys):
    rc = main(["--demo", "clean", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "clean" in out


def test_demo_buggy_text_output(capsys):
    rc = main(["--demo", "buggy", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "leak_detected" in out


def test_demo_clean_json_output(capsys):
    rc = main(["--demo", "clean", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "clean"
    assert payload["first_leak_index"] is None


def test_demo_buggy_json_output(capsys):
    rc = main(["--demo", "buggy", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "leak_detected"
    assert payload["first_leak_index"] == 0


def test_check_clean_exit_code_on_clean(capsys):
    rc = main(["--demo", "clean", "--check-clean", "--json"])
    assert rc == 0


def test_check_clean_exit_code_on_leak(capsys):
    rc = main(["--demo", "buggy", "--check-clean", "--json"])
    assert rc == 1


def test_epsilon_sweep_json(capsys):
    rc = main(["--demo", "buggy", "--json", "--epsilon-sweep"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "epsilon_sweep" in payload
    assert payload["epsilon_sweep"]["verdict"] in (
        "likely_genuine_leak",
        "likely_numerical_floor",
        "inconclusive_too_few_nonzero_points",
    )


def test_custom_seq_len_and_chunk_size(capsys):
    rc = main(["--demo", "buggy", "--json", "--seq-len", "20", "--chunk-size", "5"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "leak_detected"
