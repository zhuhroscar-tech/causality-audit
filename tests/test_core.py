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
