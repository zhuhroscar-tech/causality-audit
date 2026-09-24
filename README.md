# causality-audit

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

Check whether changing the last input position changes earlier outputs in a sequence-model layer stack. This NumPy-based prefix-invariance audit runs two forward passes, compares intermediate outputs, and identifies the first layer that exceeds a chosen tolerance.

It includes clean and deliberately buggy synthetic chunked-recurrence models for testing your audit harness. These are teaching and regression-test oracles, **not real Hugging Face checkpoints or reproductions of published benchmark numbers**.

## Install and run

Requires Python 3.9+ and NumPy 1.24+; PyTorch is not required.

```bash
git clone https://github.com/zhuhroscar-tech/causality-audit.git
cd causality-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
causality-audit --demo clean
causality-audit --demo buggy --json --epsilon-sweep
```

Add `--check-clean` to exit with code `1` when a leak is detected. Without that flag, the demo reports findings without using the exit status as a leak signal.

## Python API

```python
import numpy as np
from causality_audit.core import audit_prefix_invariance

x = np.random.default_rng(0).standard_normal((12, 3))
y = x.copy()
y[-1] += 1.0
layers = [("causal_sum", lambda a: np.cumsum(a, axis=0))]
result = audit_prefix_invariance(layers, x, y)
print(result.verdict, result.first_leak_index)
```

Supply ordered `(name, callable)` pairs. Each callable consumes the previous layer's `(seq_len, dim)` output and returns the same-shaped array. Inputs must differ only at the final position. Results include per-layer differences and `clean`, `leak_detected`, or `inconclusive` verdicts.

## Interpretation and limits

- A clean result covers the selected probe, not all possible inputs. Use a known-leaky positive control and confirm that your model responds to changed input.
- Test sequences longer than the relevant chunk/window size; a single chunk can hide cross-chunk leakage.
- Choose a tolerance appropriate for numerical precision. `epsilon_sweep` helps compare perturbation-dependent leakage with a noise floor; it is a heuristic, not proof.
- Real model adapters, forward hooks, state resets, and deterministic inference are your responsibility. This project does not validate downloaded checkpoints.
- Each layer's output shape must be identical across both forward passes. A data-dependent layer (token pruning, early-exit, sparse/MoE routing, dynamic pooling) whose output shape changes with the input is reported as `inconclusive` at that layer, not silently mis-audited or crashed on.

Background: [prefix-invariance paper](https://arxiv.org/abs/2608.22876) and [Transformers #46741](https://github.com/huggingface/transformers/pull/46741).

## Development

```bash
pytest -v --cov=causality_audit
```

See [tests](tests) for preconditions, localization, synthetic controls, and CLI behavior. Release history lives in [CHANGELOG.md](CHANGELOG.md). [MIT license](LICENSE).
