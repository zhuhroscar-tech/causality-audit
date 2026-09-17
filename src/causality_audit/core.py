"""Core audit primitive: two forward passes, compare prefixes, localize
the first layer where prefix invariance breaks.

Definition (prefix invariance): for a sequence model that consumes
x[0..T-1] and produces per-position hidden states h[0..T-1] at some
layer, prefix invariance requires that h[0..T-2] depends only on
x[0..T-2] -- i.e. changing x[T-1] alone must never change h[t] for any
t < T-1. This is the causal-language-modeling contract every autoregressive
model claims to satisfy.

The audit does not require gradients, training, or labels: it runs the
SAME model twice, with inputs that are identical except at the final
position, hooks every layer's output, and reports the first layer index
where the prefix positions (all but the last) differ beyond a numerical
noise floor `threshold`. This mirrors arXiv:2608.22876's "two forward
passes, no training or gradients" audit design, but is implemented here
as a small, dependency-free primitive over arbitrary Python callables
rather than a framework-specific model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - numpy is a hard dependency
    raise ImportError("causality_audit requires numpy") from exc

Layer = Callable[["np.ndarray"], "np.ndarray"]


@dataclass
class LayerReport:
    """Per-layer comparison result between two forward passes."""

    index: int
    name: str
    max_abs_diff_prefix: float
    max_abs_diff_last: float
    leaked: bool


@dataclass
class AuditResult:
    """Full audit result across all layers of one probe."""

    layers: List[LayerReport] = field(default_factory=list)
    first_leak_index: int | None = None
    verdict: str = "clean"  # "clean" | "leak_detected" | "inconclusive"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "first_leak_index": self.first_leak_index,
            "note": self.note,
            "layers": [
                {
                    "index": layer.index,
                    "name": layer.name,
                    "max_abs_diff_prefix": layer.max_abs_diff_prefix,
                    "max_abs_diff_last": layer.max_abs_diff_last,
                    "leaked": layer.leaked,
                }
                for layer in self.layers
            ],
        }


def run_layers(layers: Sequence[Tuple[str, Layer]], x: "np.ndarray") -> List["np.ndarray"]:
    """Run `x` through each (name, layer_fn) in sequence, collecting the
    output of every layer (not just the final one) so the audit can
    localize a leak to a specific layer index."""
    outputs = []
    current = x
    for _name, fn in layers:
        current = fn(current)
        outputs.append(current)
    return outputs


def audit_prefix_invariance(
    layers: Sequence[Tuple[str, Layer]],
    x: "np.ndarray",
    x_perturbed: "np.ndarray",
    threshold: float = 1e-6,
) -> AuditResult:
    """Run the two-forward-pass prefix-invariance audit.

    Parameters
    ----------
    layers: ordered (name, callable) pairs. Each callable takes a
        (seq_len, dim) array and returns a (seq_len, dim) array (the
        layer's own hidden-state output). Layers are applied in sequence,
        each consuming the previous layer's *output* as its input, so
        this composes like a real per-layer forward pass. If a layer
        needs the *original* input (e.g. a positional embedding), close
        over it in the callable rather than threading it through.
    x, x_perturbed: two inputs of identical shape (seq_len, dim) that
        differ ONLY at the last position (index -1). This is the
        caller's responsibility; the audit does not enforce it, but a
        violated precondition invalidates the result (see `note`).
    threshold: absolute-difference floor above which a prefix difference
        is treated as a genuine leak rather than ordinary floating-point
        noise. Default 1e-6 is appropriate for float64 reference math;
        tighten or loosen per the numeric precision of the layers under
        test.

    Returns
    -------
    AuditResult with a per-layer report and the first layer index (if
    any) where the prefix (all positions except the last) differs by
    more than `threshold` between the two runs -- i.e. where information
    from the perturbed final position leaked backward into positions
    that must not depend on it.
    """
    if x.shape != x_perturbed.shape:
        return AuditResult(verdict="inconclusive", note="shape mismatch between x and x_perturbed")
    if x.shape[0] < 2:
        return AuditResult(verdict="inconclusive", note="sequence length must be >= 2 to have a prefix")

    prefix_diff_at_input = float(np.max(np.abs(x[:-1] - x_perturbed[:-1])))
    if prefix_diff_at_input > threshold:
        return AuditResult(
            verdict="inconclusive",
            note=(
                "precondition violated: x and x_perturbed already differ on the "
                f"prefix at the input (max abs diff {prefix_diff_at_input:.3e}); "
                "the audit requires inputs identical except at the last position"
            ),
        )

    outs_a = run_layers(layers, x)
    outs_b = run_layers(layers, x_perturbed)

    result = AuditResult()
    for i, ((name, _fn), out_a, out_b) in enumerate(zip(layers, outs_a, outs_b)):
        out_a_arr = np.asarray(out_a)
        out_b_arr = np.asarray(out_b)
        if out_a_arr.shape != out_b_arr.shape:
            result.verdict = "inconclusive"
            result.note = (
                f"layer {i} ('{name}') produced different output shapes between "
                f"the two forward passes ({out_a_arr.shape} vs {out_b_arr.shape}). "
                "This audit assumes every layer's output shape is a deterministic "
                "function of the input shape alone; a data-dependent layer (token "
                "pruning, early-exit, sparse/MoE routing, dynamic pooling) breaks "
                "that assumption and cannot be compared position-by-position. "
                "Only layers before this one were audited."
            )
            return result
        prefix_a, prefix_b = out_a_arr[:-1], out_b_arr[:-1]
        last_a, last_b = out_a_arr[-1], out_b_arr[-1]
        max_diff_prefix = float(np.max(np.abs(prefix_a - prefix_b))) if prefix_a.size else 0.0
        max_diff_last = float(np.max(np.abs(last_a - last_b))) if np.size(last_a) else 0.0
        leaked = max_diff_prefix > threshold
        report = LayerReport(
            index=i,
            name=name,
            max_abs_diff_prefix=max_diff_prefix,
            max_abs_diff_last=max_diff_last,
            leaked=leaked,
        )
        result.layers.append(report)
        if leaked and result.first_leak_index is None:
            result.first_leak_index = i

    if result.first_leak_index is not None:
        result.verdict = "leak_detected"
        result.note = (
            f"prefix diverges starting at layer {result.first_leak_index} "
            f"('{result.layers[result.first_leak_index].name}'): perturbing only "
            "the final input position changed a prefix position's output, which "
            "violates prefix invariance (a causality leak)"
        )
    else:
        result.verdict = "clean"
        result.note = "no layer showed a prefix difference above threshold"
    return result


def epsilon_sweep(
    build_layers: Callable[[], Sequence[Tuple[str, Layer]]],
    x_base: "np.ndarray",
    epsilons: Sequence[float],
    probe_layer_index: int,
    threshold: float = 1e-6,
) -> dict:
    """Discriminate a genuine leak from a numerical noise floor by
    scaling the perturbation magnitude at the last position and checking
    whether the detected prefix divergence climbs with it (paper's
    "epsilon-sweep discriminator", Figure 2 of arXiv:2608.22876: real
    leaks show a positive log-log slope; numerical floors stay flat).

    Returns a dict with the raw (epsilon, max_abs_diff_prefix) pairs at
    `probe_layer_index` and a fitted log-log slope. A slope reliably
    above ~0.1 with low residual is evidence of a genuine leak; a slope
    near 0 (flat) indicates a numerical floor, not a structural leak.
    """
    points = []
    for eps in epsilons:
        layers = build_layers()
        x_perturbed = x_base.copy()
        x_perturbed[-1] += eps
        result = audit_prefix_invariance(layers, x_base, x_perturbed, threshold=threshold)
        if probe_layer_index < len(result.layers):
            diff = result.layers[probe_layer_index].max_abs_diff_prefix
        else:
            diff = 0.0
        points.append((eps, diff))

    xs = [math.log(abs(e)) for e, d in points if e != 0 and d > 0]
    ys = [math.log(d) for e, d in points if e != 0 and d > 0]
    slope = None
    if len(xs) >= 2:
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var = sum((x - mean_x) ** 2 for x in xs)
        slope = cov / var if var > 0 else None

    return {
        "points": points,
        "slope": slope,
        "verdict": (
            "likely_genuine_leak" if slope is not None and slope > 0.1
            else "likely_numerical_floor" if slope is not None
            else "inconclusive_too_few_nonzero_points"
        ),
    }
