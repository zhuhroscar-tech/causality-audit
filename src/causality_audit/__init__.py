"""causality-audit: a lightweight, two-forward-pass audit for prefix
invariance (causal-leakage) violations in sequence models.

Based on the finding in arXiv:2608.22876 ("The Mask Is Not the Model:
Auditing Prefix Invariance in Attention, State-Space, and Hybrid Sequence
Models", VIDRAFT AI Research, 2026-08-24): attention-mask inspection is
structurally blind to causality violations that occur inside a scan or a
normalization step rather than inside the attention operator itself. The
paper demonstrates this concretely against real, released chunked-scan
implementations (Zamba2, Nemotron-H) where the inter-chunk recurrence
reduces over the wrong chunk axis, letting information from later chunks
leak into earlier positions -- a bug independently confirmed and fixed
upstream (huggingface/transformers#46741, building on #46032/#46084).

This package does NOT bundle or require torch/transformers. It provides:

1. A generic, dependency-free audit primitive (`audit_prefix_invariance`)
   that works on any ordered sequence of hookable "layers" -- pure Python
   callables, numpy arrays in, numpy arrays out.
2. A deterministic toy reference implementation of a correct vs.
   axis-swapped chunked linear recurrence, reproducing the *bug class*
   from the paper (not a claim about the paper's own numbers), used as
   this project's test oracle.

See README.md for exact scope and limitations -- in particular, this
project has not audited any real HuggingFace checkpoint; that requires
torch+transformers and is documented as a manual, non-CI workflow.
"""

__version__ = "0.2.4"
