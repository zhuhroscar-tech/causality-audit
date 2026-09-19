"""Tests for the core prefix-invariance audit primitive."""
from __future__ import annotations

import numpy as np
import pytest

from causality_audit.core import LayerReport, audit_prefix_invariance, epsilon_sweep


def _identity_layers(n=2):
    return [(f"identity_{i}", lambda x: x.copy()) for i in range(n)]


def _leaky_layer_at(index, n=3):
    """Build n layers where layer `index` deliberately leaks: it
    overwrites every prefix position with the last position's value,
    an obvious, unambiguous violation used to test localization."""
    layers = []
    for i in range(n):
        if i == index:
            def fn(x):
                out = x.copy()
                out[:-1] = x[-1]
                return out
            layers.append((f"leaky_{i}", fn))
        else:
            layers.append((f"pass_{i}", lambda x: x.copy()))
    return layers


class TestAuditPreconditions:
    def test_rejects_shape_mismatch(self):
        x = np.zeros((5, 2))
        x2 = np.zeros((6, 2))
        result = audit_prefix_invariance(_identity_layers(), x, x2)
        assert result.verdict == "inconclusive"
        assert "shape mismatch" in result.note

    def test_rejects_too_short_sequence(self):
        x = np.zeros((1, 2))
        result = audit_prefix_invariance(_identity_layers(), x, x.copy())
        assert result.verdict == "inconclusive"
        assert "prefix" in result.note

    def test_rejects_precondition_violation_prefix_already_differs(self):
        x = np.zeros((4, 2))
        x2 = x.copy()
        x2[0] += 1.0  # violates "identical except at last position"
        result = audit_prefix_invariance(_identity_layers(), x, x2)
        assert result.verdict == "inconclusive"
        assert "precondition violated" in result.note


class TestAuditLayerOutputShapeMismatch:
    """A layer whose OUTPUT shape depends on the data (token pruning,
    early-exit, sparse/MoE routing, dynamic pooling) is a realistic
    pattern in real sequence models. The precondition check only
    validates the two *inputs'* shapes match; nothing validated that a
    layer's output shape stays identical across both forward passes.
    When it silently doesn't, the raw numpy subtraction two lines later
    either raises an unhandled ValueError (unequal leading dims) or,
    worse, silently broadcasts/truncates instead of reporting a clear
    'inconclusive' precondition violation like every other invalid-input
    case in this module already does."""

    def test_layer_output_shape_mismatch_is_reported_not_crashed(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal((6, 3))
        x2 = x.copy()
        x2[-1] += 5.0

        def data_dependent_layer(a):
            # Prunes rows above a magnitude threshold -- output row count
            # depends on the perturbed value, so the two passes' outputs
            # end up with DIFFERENT leading shapes.
            mask = np.abs(a[:, 0]) < 3.0
            return a[mask]

        layers = [("prune", data_dependent_layer)]
        result = audit_prefix_invariance(layers, x, x2)
        assert result.verdict == "inconclusive"
        assert "prune" in result.note
        assert "shape" in result.note.lower()

    def test_layer_output_shape_mismatch_localizes_to_the_right_layer(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal((6, 3))
        x2 = x.copy()
        x2[-1] += 5.0

        def identity(a):
            return a.copy()

        def shrink(a):
            return a[: a.shape[0] - int(a[-1, 0] > 0)]

        layers = [("pass0", identity), ("shrink1", shrink)]
        result = audit_prefix_invariance(layers, x, x2)
        assert result.verdict == "inconclusive"
        assert "shrink1" in result.note
        # layer 0 should still have a valid, non-crashing report
        assert len(result.layers) == 1
        assert result.layers[0].name == "pass0"


class TestAuditNaNHandling:
    """A layer that produces NaN in a prefix position (numerical blowup,
    a real leak that happens to divide-by-zero, masked/padded regions,
    etc.) must never be silently reported as 'clean'. `np.max(np.abs(diff))`
    over an array containing NaN returns NaN, and `NaN > threshold` is
    always False in numpy/Python -- so before this fix, any NaN in a
    layer's prefix output made `leaked` False and the overall verdict
    'clean', which is the worst possible failure mode for a leak
    detector: it actively hides the most suspicious case behind its
    most reassuring verdict."""

    def test_nan_in_layer_prefix_output_is_not_reported_clean(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((4, 2))
        x2 = x.copy()
        x2[-1] += 1.0

        def nan_injector(a):
            out = a.copy()
            out[0, 0] = np.nan  # corrupt a prefix position's output
            return out

        result = audit_prefix_invariance([("nan_layer", nan_injector)], x, x2)
        assert result.verdict != "clean"
        assert result.verdict == "inconclusive"
        assert "nan" in result.note.lower()

    def test_nan_precondition_violation_is_not_reported_clean(self):
        # The precondition check itself must also not be fooled by NaN:
        # x and x_perturbed disagreeing on the prefix via a NaN (instead
        # of an ordinary numeric difference) must still be caught.
        x = np.array([[1.0], [2.0], [3.0]])
        x2 = np.array([[1.0], [np.nan], [4.0]])
        result = audit_prefix_invariance([("identity", lambda a: a.copy())], x, x2)
        assert result.verdict == "inconclusive"
        assert "nan" in result.note.lower()

    def test_to_dict_still_serializable_with_nan_verdict(self):
        import json

        x = np.zeros((3, 1))
        x2 = x.copy()
        x2[-1] += 1.0

        def nan_injector(a):
            out = a.copy()
            out[0, 0] = np.nan
            return out

        result = audit_prefix_invariance([("nan_layer", nan_injector)], x, x2)
        json.dumps(result.to_dict())  # must not raise


class TestAuditDetection:
    def test_identity_layers_report_clean(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((6, 3))
        x2 = x.copy()
        x2[-1] += 5.0
        result = audit_prefix_invariance(_identity_layers(), x, x2)
        assert result.verdict == "clean"
        assert result.first_leak_index is None
        assert all(not layer.leaked for layer in result.layers)

    def test_leak_detected_and_localized_to_exact_layer(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((6, 3))
        x2 = x.copy()
        x2[-1] += 5.0
        for leak_index in (0, 1, 2):
            layers = _leaky_layer_at(leak_index, n=3)
            result = audit_prefix_invariance(layers, x, x2)
            assert result.verdict == "leak_detected", f"expected leak at {leak_index}"
            assert result.first_leak_index == leak_index

    def test_reports_all_layer_indices_and_names(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((5, 2))
        x2 = x.copy()
        x2[-1] += 1.0
        layers = _identity_layers(4)
        result = audit_prefix_invariance(layers, x, x2)
        assert len(result.layers) == 4
        for i, layer in enumerate(result.layers):
            assert isinstance(layer, LayerReport)
            assert layer.index == i
            assert layer.name == f"identity_{i}"

    def test_to_dict_is_json_serializable(self):
        import json

        rng = np.random.default_rng(3)
        x = rng.standard_normal((5, 2))
        x2 = x.copy()
        x2[-1] += 1.0
        result = audit_prefix_invariance(_leaky_layer_at(1), x, x2)
        json.dumps(result.to_dict())  # must not raise


class TestEpsilonSweep:
    def test_genuine_leak_shows_positive_slope(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((6, 3))

        def build():
            return _leaky_layer_at(0, n=1)

        sweep = epsilon_sweep(build, x, epsilons=[1e-3, 1e-2, 1e-1, 1.0], probe_layer_index=0, threshold=1e-12)
        assert sweep["slope"] is not None
        assert sweep["slope"] > 0.5  # leaky_layer copies last value exactly: slope should be ~1.0
        assert sweep["verdict"] == "likely_genuine_leak"

    def test_numerical_floor_shows_flat_slope(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((6, 3))

        # A "layer" whose output is completely insensitive to input
        # magnitude beyond a tiny constant floor -- models a numerical
        # floor, not a structural leak.
        def build():
            return [("floor_layer", lambda arr: np.full_like(arr, 1e-10))]

        sweep = epsilon_sweep(build, x, epsilons=[1e-3, 1e-2, 1e-1, 1.0], probe_layer_index=0, threshold=1e-15)
        # constant output regardless of epsilon -> diff is ~0 for the prefix in all cases (no leak at all)
        assert sweep["verdict"] in ("likely_numerical_floor", "inconclusive_too_few_nonzero_points")

    def test_out_of_range_probe_layer_index_defaults_diff_to_zero(self):
        """probe_layer_index >= len(result.layers) must fall through to the
        diff = 0.0 else-branch (core.py line 234), not IndexError. This
        happens for real when a caller passes a stale/max first_leak_index
        from a differently-shaped audit result (e.g. cli.py's
        `probe_index = result.first_leak_index if ... else 0` computed
        against one model, then reused after a config change that shrinks
        the layer count)."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal((6, 3))

        def build():
            return _leaky_layer_at(0, n=1)

        # This reference model has exactly 1 layer (index 0 valid, index 5
        # is out of range) -- forces every point through the else-branch.
        sweep = epsilon_sweep(build, x, epsilons=[1e-3, 1e-2, 1e-1, 1.0], probe_layer_index=5, threshold=1e-12)
        assert all(diff == 0.0 for _eps, diff in sweep["points"])
        assert sweep["verdict"] == "inconclusive_too_few_nonzero_points"
