"""A minimal, deterministic toy chunked linear-recurrence layer, built to
reproduce the *shape* of the real bug documented in arXiv:2608.22876 and
independently confirmed/fixed upstream in huggingface/transformers#46741
(Zamba2 / Nemotron-H's Mamba2-style mixer): an inter-chunk recurrence
that reduces over the wrong chunk axis lets a later chunk's input leak
into an earlier chunk's carried-forward state.

This is NOT a reimplementation of Mamba2/Zamba2/Nemotron-H itself (no
torch, no real model weights, no SSM selectivity) -- it is a small
synthetic construction, built from scratch for this project, whose
*fault-injection structure* mirrors the paper's diagnosis:

  - Sequence is split into fixed-size chunks.
  - Within a chunk, the recurrence is a straightforward causal
    (lower-triangular) linear scan -- correct by construction.
  - Between chunks, each chunk's carry-in state should depend only on
    chunks *before* it (a causal reduction over the chunk axis).
  - The injected bug reduces over the *output* chunk axis instead of the
    *input* chunk axis (mirroring the paper's root-cause description:
    "the reduction runs over the output-chunk axis rather than the
    input-chunk axis"), which lets a later chunk's contribution leak
    into an earlier chunk's carry-in.

The correct and buggy variants are otherwise byte-identical: same shapes,
same decay/state update math within a chunk, differing only in the one
transpose/axis choice for the inter-chunk reduction, exactly mirroring
the paper's characterization of the real defect as "three lines."
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def _within_chunk_scan(chunk: np.ndarray, decay: float) -> np.ndarray:
    """A correct, causal (lower-triangular) exponential-decay scan within
    one chunk: state[t] = decay * state[t-1] + chunk[t]. Deliberately
    simple and obviously causal -- the bug this project targets never
    lives here, it lives in the inter-chunk reduction below."""
    T = chunk.shape[0]
    out = np.zeros_like(chunk)
    state = np.zeros(chunk.shape[1:], dtype=chunk.dtype)
    for t in range(T):
        state = decay * state + chunk[t]
        out[t] = state
    return out


def _chunk_carry_ins_correct(chunks: List[np.ndarray], decay: float) -> List[np.ndarray]:
    """Correct inter-chunk carry-in: chunk i's carry-in state is the
    decayed cumulative sum of chunks 0..i-1 only (causal over the chunk
    axis -- the INPUT chunk axis). Chunk 0's carry-in is zero."""
    carries = [np.zeros(chunks[0].shape[1:], dtype=chunks[0].dtype)]
    running = np.zeros(chunks[0].shape[1:], dtype=chunks[0].dtype)
    for chunk in chunks[:-1]:
        chunk_end_state = _within_chunk_scan(chunk, decay)[-1]
        running = decay * running + chunk_end_state
        carries.append(running.copy())
    return carries


def _chunk_carry_ins_buggy(chunks: List[np.ndarray], decay: float) -> List[np.ndarray]:
    """Buggy inter-chunk carry-in: reduces over the OUTPUT chunk axis
    instead of the input chunk axis. Concretely, the correct code takes
    an EXCLUSIVE prefix (cumulative) sum of end-states in chunk order --
    carry-in[i] depends only on chunks strictly before i. The injected
    bug instead computes an INCLUSIVE SUFFIX sum (i.e. the cumulative
    reduction is taken over the reversed/output-indexed axis rather than
    the forward input-indexed axis), so carry-in[i] for i >= 1 picks up
    contributions from chunk i itself and every later chunk -- exactly
    the shape of leak the paper describes ("information from later
    positions reaches the carry-in used to recompute earlier positions'
    outputs").

    Chunk 0's carry-in is still forced to zero, matching the paper's own
    observation that both real buggy implementations "prepend a zero
    carry state... before the recurrence," which is why the leak's exact
    onset is the first chunk *boundary*, not position 0."""
    n = len(chunks)
    if n <= 1:
        return [np.zeros(chunks[0].shape[1:], dtype=chunks[0].dtype)]
    end_states = np.stack([_within_chunk_scan(c, decay)[-1] for c in chunks])
    # Correct code: exclusive prefix cumsum over the INPUT chunk axis
    # (forward order). Bug: inclusive suffix cumsum -- the reduction
    # runs over the reversed (OUTPUT) chunk axis instead.
    reversed_inclusive = np.cumsum(end_states[::-1], axis=0)[::-1]
    carries = [np.zeros_like(end_states[0])]
    for i in range(1, n):
        carries.append(decay * reversed_inclusive[i])
    return carries


def chunked_scan_layer(chunk_size: int, decay: float, buggy: bool) -> Callable[[np.ndarray], np.ndarray]:
    """Build a layer callable (seq_len, dim) -> (seq_len, dim) implementing
    a chunked causal scan, correct or with the injected axis-swap bug."""

    def layer(x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        chunks = [x[i : i + chunk_size] for i in range(0, T, chunk_size)]
        carry_fn = _chunk_carry_ins_buggy if buggy else _chunk_carry_ins_correct
        carries = carry_fn(chunks, decay)
        outputs = []
        for chunk, carry_in in zip(chunks, carries):
            local = _within_chunk_scan(chunk, decay)
            outputs.append(local + carry_in)
        return np.concatenate(outputs, axis=0)

    return layer


def build_reference_model(
    chunk_size: int = 4,
    decay: float = 0.9,
    buggy: bool = False,
    n_layers: int = 2,
) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    """A tiny multi-layer "model": each layer is the same chunked-scan
    primitive (correct or buggy per `buggy`), which is enough to exercise
    the audit's per-layer localization without needing a real model."""
    return [
        (f"chunked_scan_layer_{i}", chunked_scan_layer(chunk_size, decay, buggy))
        for i in range(n_layers)
    ]
