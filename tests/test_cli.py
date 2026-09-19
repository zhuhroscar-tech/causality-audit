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


@pytest.mark.parametrize("chunk_size", ["0", "-1", "-5"])
def test_nonpositive_chunk_size_is_clean_cli_error_not_crash(capsys, chunk_size):
    """--chunk-size <= 0 must produce a clean argparse usage error (exit 2),
    never an unhandled ValueError/IndexError traceback from reference.py's
    range()/list-indexing internals leaking past the CLI boundary."""
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "clean", "--chunk-size", chunk_size])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "chunk-size" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("seq_len", ["0", "-1"])
def test_nonpositive_seq_len_is_clean_cli_error_not_crash(capsys, seq_len):
    """--seq-len <= 0 must produce a clean argparse usage error (exit 2),
    never an unhandled IndexError from indexing an empty/negative array."""
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "clean", "--seq-len", seq_len])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "seq-len" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("flag,value", [("--seq-len", "abc"), ("--chunk-size", "3.5")])
def test_non_integer_positive_int_flags_are_clean_cli_error_not_crash(capsys, flag, value):
    """--seq-len/--chunk-size given a non-integer string (e.g. 'abc' or a
    float-looking '3.5') must hit the int(value) ValueError branch in
    _positive_int()'s _parse() and surface as a clean argparse usage error
    (exit 2, 'must be an integer' in stderr), never an uncaught traceback.
    This exercises cli.py's _parse() ValueError->ArgumentTypeError path
    (lines 26-27), which every prior test skipped by only ever supplying
    valid-looking or non-positive-but-still-integer strings."""
    with pytest.raises(SystemExit) as exc:
        main(["--demo", "clean", flag, value])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "must be an integer" in err
    assert "Traceback" not in err


def test_epsilon_sweep_text_output(capsys):
    """--epsilon-sweep without --json must render the human-readable
    'Epsilon-sweep discriminator' section (cli.py main()'s text-output
    branch, lines 136-141), not just the --json branch that every prior
    epsilon-sweep test exercised."""
    rc = main(["--demo", "buggy", "--epsilon-sweep", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Epsilon-sweep discriminator" in out
    assert "slope=" in out
    assert "eps=" in out
    assert "max_abs_diff_prefix=" in out
