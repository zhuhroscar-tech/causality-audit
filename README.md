# causality-audit

A lightweight, dependency-free-on-torch audit primitive for **prefix
invariance** (causal-leakage) violations in sequence-model layers, plus a
synthetic reference/oracle that reproduces a real, documented bug class.

## The problem

Autoregressive sequence models (Transformers, SSMs like Mamba, and hybrids
like Zamba2/Nemotron-H) all claim the same causal contract: the
representation at position `t` must never depend on inputs after `t`. The
standard way people check this is inspecting the attention mask. That check
is **structurally incomplete**: a scan (state-space recurrence) or a
normalization step has no mask at all, so a causality violation there is
invisible to mask inspection even when the mask itself is perfectly
correct.

arXiv:2608.22876 ("The Mask Is Not the Model: Auditing Prefix Invariance in
Attention, State-Space, and Hybrid Sequence Models", VIDRAFT AI Research,
2026-08-24) demonstrates this concretely: across 192 injected-fault trials
on 8 real checkpoints, attention-mask inspection caught **0**, while a
two-forward-pass audit (feed two inputs identical except at the last
position, hook every layer, find where the prefix first diverges) localized
**192/192**. The same audit predicted from source code alone, and then
confirmed dynamically, that Zamba2-1.2B and Nemotron-H-8B's Mamba2-style
chunked-scan implementations leak information across chunk boundaries. This
is independently corroborated by a real, merged upstream fix:
[huggingface/transformers#46741](https://github.com/huggingface/transformers/pull/46741)
(building on #46032/#46084), which fixes the same class of chunked-scan
causality bug in Zamba2, Nemotron-H, Bamba, FalconH1, and GraniteMoeHybrid,
with measured before/after diffs (e.g. Zamba2: 1.2e-1 → 5.96e-8).

At the time this project was built, no released open-source tool
implemented this audit as a standalone, reusable primitive (the paper
itself states no code was released alongside it).

## What this tool actually does (and does not)

- **Does**: provide `audit_prefix_invariance()`, a small, numpy-only
  primitive that runs two forward passes over an arbitrary ordered sequence
  of Python callables ("layers"), each `(seq_len, dim) -> (seq_len, dim)`,
  and reports the first layer index where a prefix position's output
  changes because of a perturbation that should only affect the last
  position. No gradients, no training, CPU-only, runs in milliseconds for
  the included examples.
- **Does**: ship a synthetic, from-scratch chunked linear-recurrence
  reference model (`causality_audit.reference`) whose injected bug mirrors
  the *causal structure* of the real Zamba2/Nemotron-H defect (a reduction
  over the wrong chunk axis) — used as this project's test oracle. This is
  **not** a reimplementation of Mamba2/Zamba2/Nemotron-H; it has no SSM
  selectivity, no real model weights, and is not claimed to reproduce the
  paper's own numerical results.
- **Does not** audit any real HuggingFace checkpoint out of the box. Doing
  that requires `torch` + `transformers` plus forward hooks wired to each
  model's actual module tree — deliberately kept out of this package's
  hard dependencies so the core audit stays framework-agnostic and
  installable anywhere. Wiring a real model's layers into
  `audit_prefix_invariance()` is a few lines (see "Auditing a real model"
  below) but has not itself been run against a downloaded checkpoint by
  this project — that is a stated limitation, not a hidden one.
- **Does** include an epsilon-sweep discriminator
  (`causality_audit.core.epsilon_sweep`) to distinguish a genuine
  structural leak (log-log slope reliably > 0 as perturbation magnitude
  grows) from an ordinary floating-point noise floor (flat slope), mirroring
  the paper's own Figure 2 discriminator.

## Install

```bash
pip install causality-audit
```

## Usage

```bash
# Run the built-in synthetic reference model (correct variant) and audit it
causality-audit --demo clean

# Run the built-in buggy variant (injected chunk-axis-swap bug)
causality-audit --demo buggy

# Machine-readable JSON
causality-audit --demo buggy --json

# Also run the epsilon-sweep discriminator
causality-audit --demo buggy --epsilon-sweep

# CI-friendly: exit 1 if the audited model shows a leak
causality-audit --demo buggy --check-clean
```

### Auditing a real model (not covered by this project's own tests)

```python
import numpy as np
from causality_audit.core import audit_prefix_invariance

# Build (name, callable) pairs from your own model's layers, e.g. via
# forward hooks that capture each decoder layer's output as a numpy array.
layers = [("layer_0", my_layer_0_fn), ("layer_1", my_layer_1_fn), ...]

x = np.random.default_rng(0).standard_normal((seq_len, hidden_dim))
x_perturbed = x.copy()
x_perturbed[-1] += 1.0  # perturb ONLY the last position

result = audit_prefix_invariance(layers, x, x_perturbed)
print(result.verdict, result.first_leak_index, result.note)
```

Two things the paper found and this tool inherits as caveats:

1. **Audit length must exceed the model's chunk/window size.** A leak that
   only manifests once a second chunk exists will look clean if you audit
   a sequence shorter than the chunk size (see
   `test_single_chunk_sequence_is_clean_even_when_buggy` in this repo's
   tests — this is expected, not a bug in the audit).
2. **A clean verdict needs a positive control.** If a checkpoint returns
   bit-identical outputs regardless of input (a degenerate/untrained/
   misconfigured model), the audit will report `clean` — correctly, but
   uninformatively. Verify your harness first with a known-leaky case
   (this project's `--demo buggy` is one) before trusting a `clean` result
   on a real model.

## Verification

- `pytest --cov` on this repo: the core audit primitive (preconditions,
  clean detection, exact-layer localization, epsilon-sweep discriminator),
  the synthetic reference oracle (correct variant never leaks across
  multiple sequence lengths, buggy variant always leaks at the expected
  layer, single-chunk sequences correctly look clean even when buggy — a
  regression test for the paper's own "audit length must exceed chunk
  size" caveat), and the CLI (JSON output, `--no-color`, `--check-clean`
  exit codes).
- CI runs on both `ubuntu-latest` and `macos-latest` GitHub Actions runners
  across Python 3.9 and 3.12. All logic here is pure Python + numpy, so
  this cross-platform CI genuinely exercises the real code path on both
  OSes (unlike a claim of Linux/macOS parity based on local testing alone).
- Released wheels/sdists are checksummed (`SHA256SUMS.txt`) and smoke
  tested from a clean venv as part of CI.
- **Not verified**: this project has not been run against any real
  HuggingFace checkpoint (Zamba2, Nemotron-H, or otherwise) — that would
  require `torch`+`transformers` and multi-gigabyte model downloads, kept
  out of scope for this package and its CI. The synthetic reference model
  is a from-scratch construction whose *fault-injection structure* mirrors
  the paper's diagnosis; it is not a validated reproduction of the paper's
  own reported numbers.

## License

MIT
