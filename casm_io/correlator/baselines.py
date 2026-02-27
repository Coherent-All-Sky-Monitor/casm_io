"""
Upper-triangular baseline indexing for visibility matrices.

CASM visibility data is stored as a flattened upper-triangular matrix
(including diagonal for autocorrelations). This module provides utilities
for indexing into this structure.
"""

import numpy as np


def triu_flat_index(n: int, i: int, j: int) -> int:
    """
    Index into flattened upper-triangular (including diagonal).

    Ordering matches np.triu_indices(n): row 0 cols 0..n-1,
    row 1 cols 1..n-1, etc.

    Parameters
    ----------
    n : int
        Matrix dimension (number of inputs).
    i : int
        Row index (must be <= j).
    j : int
        Column index (must be >= i).

    Returns
    -------
    int
        Flat index into the triangular vector.
    """
    if i > j:
        raise ValueError(f"Expected i <= j, got i={i}, j={j}")
    base = i * n - (i * (i - 1)) // 2
    return base + (j - i)


def triu_to_ij(n: int, flat_idx: int) -> tuple[int, int]:
    """Convert flat index back to (i, j) pair where i <= j."""
    i = 0
    remaining = flat_idx
    while remaining >= (n - i):
        remaining -= (n - i)
        i += 1
    return i, i + remaining


def n_baselines(nsig: int) -> int:
    """Number of baselines including autos: nsig*(nsig+1)/2."""
    return nsig * (nsig + 1) // 2


def build_baseline_plan(
    ref: int, targets: list[int], nsig: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build extraction plan for ref->target baselines.

    Parameters
    ----------
    ref : int
        Reference input index.
    targets : list of int
        Target input indices.
    nsig : int
        Number of signal inputs.

    Returns
    -------
    bl_indices : np.ndarray
        Flat indices into triangular vector.
    conjugate : np.ndarray
        Boolean array, True where conjugation needed to get V(ref, target).
    """
    if not (0 <= ref < nsig):
        raise ValueError(f"ref {ref} out of range 0..{nsig - 1}")
    tlist = [int(t) for t in targets]
    if any((t < 0 or t >= nsig) for t in tlist):
        raise ValueError(f"Some targets out of range 0..{nsig - 1}")
    if ref in tlist:
        raise ValueError("targets must not include ref")
    if len(set(tlist)) != len(tlist):
        raise ValueError("Duplicate entries in targets")

    bl_indices = np.empty(len(tlist), dtype=np.int64)
    conjugate = np.empty(len(tlist), dtype=bool)

    for k, t in enumerate(tlist):
        i = min(ref, t)
        j = max(ref, t)
        bl_indices[k] = triu_flat_index(nsig, i, j)
        conjugate[k] = ref > t

    return bl_indices, conjugate


def extract_baselines(
    data: np.ndarray,
    bl_indices: np.ndarray,
    conjugate: np.ndarray,
) -> np.ndarray:
    """
    Extract and orient baselines as V(ref->target).

    Parameters
    ----------
    data : np.ndarray
        Visibility data, shape (T, F, nbaseline, 2) for real/imag
        or (T, F, nbaseline) if already complex.
    bl_indices : np.ndarray
        Baseline indices from build_baseline_plan.
    conjugate : np.ndarray
        Conjugation flags from build_baseline_plan.

    Returns
    -------
    np.ndarray
        Complex visibility array (T, F, len(bl_indices)).
    """
    if data.ndim == 4:
        sel = data[:, :, bl_indices, :]
        v = sel[..., 0] + 1j * sel[..., 1]
    else:
        v = data[:, :, bl_indices].copy()

    if np.any(conjugate):
        v[:, :, conjugate] = np.conj(v[:, :, conjugate])

    return v.astype(np.complex64)
