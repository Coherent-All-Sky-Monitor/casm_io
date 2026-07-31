"""Form visibilities from dumped voltages.

One function, one rule: multiply the inputs pairwise first, then average
over the integration time. Averaging the voltages first destroys the
signal, so this module does the ordering for you.
"""

import numpy as np

from .._results import CorrelateResult
from ..constants import TSAMP_S


def correlate(voltages, tint_s: float | None = None,
              tint_samples: int | None = None) -> CorrelateResult:
    """Visibilities from voltages: vis[t, f, i, j] = <v_i conj(v_j)>.

    Parameters
    ----------
    voltages : np.ndarray or dict
        Either a (n_time, n_chan, n_input) complex array (what
        read_full_band gives with antenna_csv), or the raw snap dict
        {snap: (n_time, n_chan, n_adc)} — the dict is stacked along the
        input axis in (snap, adc) order and the pairs come back in
        ``inputs``.
    tint_s : float, optional
        Integration time in seconds, rounded to whole samples (one sample
        is 32.768 us). Default: no averaging.
    tint_samples : int, optional
        Integration length in samples; 1 (the default) keeps the native
        time resolution. Give this or tint_s, not both.

    Returns
    -------
    CorrelateResult
        vis (n_bin, n_chan, n_input, n_input) complex64, time_s bin-centre
        times from the start of the array, tint_samples actually used, and
        the stacked (snap, adc) input labels for dict input. Samples past
        the last whole bin are dropped.
    """
    if tint_s is not None and tint_samples is not None:
        raise ValueError("give tint_s or tint_samples, not both")
    if tint_s is not None:
        tint_samples = max(int(round(tint_s / TSAMP_S)), 1)
    if tint_samples is None:
        tint_samples = 1
    if tint_samples < 1:
        raise ValueError(f"tint_samples must be >= 1, got {tint_samples}")

    inputs = None
    if isinstance(voltages, dict):
        snaps = sorted(voltages)
        v = np.concatenate([voltages[s] for s in snaps], axis=2)
        inputs = [(s, adc) for s in snaps
                  for adc in range(voltages[s].shape[2])]
    else:
        v = voltages

    n_bin = v.shape[0] // tint_samples
    if n_bin == 0:
        raise ValueError(
            f"integration of {tint_samples} samples is longer than the "
            f"{v.shape[0]} samples given"
        )

    vb = v[: n_bin * tint_samples].reshape(
        n_bin, tint_samples, v.shape[1], v.shape[2],
    )
    vis = np.einsum("tbfi,tbfj->tfij", vb, np.conj(vb)) / tint_samples
    time_s = (np.arange(n_bin) + 0.5) * tint_samples * TSAMP_S
    return CorrelateResult(vis=vis.astype(np.complex64), time_s=time_s,
                           tint_samples=tint_samples, inputs=inputs)
