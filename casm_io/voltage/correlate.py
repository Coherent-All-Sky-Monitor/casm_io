"""Form visibilities from dumped voltages.

One function, one rule: multiply the inputs pairwise first, then average
over the integration time. Averaging the voltages first destroys the
signal, so this module does the ordering for you.

The same call takes one array or a stream of gulps from
``VoltageReader.iter_full_band()``. Given a stream it returns a generator
of the same results, integration bins carried across the gulp boundaries
and time_s counted from the start of the stream, so a dump too long to
unpack at once correlates exactly like one held whole.
"""

import numpy as np

from .._results import CorrelateResult
from ..constants import TSAMP_S


def _resolve_tint(tint_s, tint_samples) -> int:
    """Integration length in whole samples from either argument."""
    if tint_s is not None and tint_samples is not None:
        raise ValueError("give tint_s or tint_samples, not both")
    if tint_s is not None:
        tint_samples = max(int(round(tint_s / TSAMP_S)), 1)
    if tint_samples is None:
        tint_samples = 1
    if tint_samples < 1:
        raise ValueError(f"tint_samples must be >= 1, got {tint_samples}")
    return tint_samples


def _as_inputs_array(voltages,
                     inputs=None) -> tuple[np.ndarray, list | None]:
    """(n_time, n_chan, n_input) array and its labels from any gulp form.

    Accepts a FullBandResult, the raw snap dict, or an array already
    stacked on the input axis. ``inputs`` keeps a subset of the input axis:
    (snap, adc) pairs against the dict, column indices against an array.
    Only the requested columns are stacked, so an unwanted ADC is never
    copied and the einsum shrinks with the square of the count.
    """
    if hasattr(voltages, "voltages"):        # FullBandResult
        voltages = voltages.voltages

    if isinstance(voltages, dict):
        if inputs is not None:
            missing = [p for p in inputs
                       if p[0] not in voltages
                       or not 0 <= p[1] < voltages[p[0]].shape[2]]
            if missing:
                raise ValueError(
                    f"(snap, adc) inputs {missing} are not in the data; "
                    f"snaps read: {sorted(voltages)}"
                )
            v = np.stack([voltages[s][:, :, adc] for s, adc in inputs],
                         axis=2)
            return v, [tuple(p) for p in inputs]
        snaps = sorted(voltages)
        v = np.concatenate([voltages[s] for s in snaps], axis=2)
        labels = [(s, adc) for s in snaps
                  for adc in range(voltages[s].shape[2])]
        return v, labels

    if inputs is not None:
        if any(isinstance(p, tuple) for p in inputs):
            raise ValueError(
                "(snap, adc) inputs need the snap dict; this read gave a "
                "stacked array (antenna_csv=), so use column indices"
            )
        bad = [i for i in inputs if not 0 <= i < voltages.shape[2]]
        if bad:
            raise ValueError(
                f"input indices {bad} are outside the "
                f"{voltages.shape[2]} columns read"
            )
        voltages = voltages[:, :, list(inputs)]
    return voltages, None


def _correlate_block(v: np.ndarray, tint_samples: int) -> np.ndarray:
    """vis (n_bin, n_chan, n_input, n_input) for a whole number of bins."""
    n_bin = v.shape[0] // tint_samples
    vb = v[: n_bin * tint_samples].reshape(
        n_bin, tint_samples, v.shape[1], v.shape[2],
    )
    vis = np.einsum("tbfi,tbfj->tfij", vb, np.conj(vb)) / tint_samples
    return vis.astype(np.complex64)


def _correlate_stream(chunks, tint_samples: int, inputs=None):
    """Correlate a stream of gulps, bins spanning the gulp boundaries.

    Samples left over at the end of a gulp are carried into the next one,
    so the visibilities do not depend on how the read was chunked. A tail
    shorter than one integration is dropped, as it is for a single array.
    """
    buf = None
    labels = None
    done = 0
    for chunk in chunks:
        v, chunk_labels = _as_inputs_array(chunk, inputs)
        if labels is None:
            labels = chunk_labels
        if buf is not None and buf.shape[0]:
            v = np.concatenate([buf, v], axis=0)
        n_bin = v.shape[0] // tint_samples
        if n_bin == 0:
            buf = v                      # not a whole bin yet, keep reading
            continue
        used = n_bin * tint_samples
        buf = v[used:].copy()            # don't pin the gulp alive
        vis = _correlate_block(v[:used], tint_samples)
        time_s = (np.arange(n_bin) + 0.5) * tint_samples * TSAMP_S
        yield CorrelateResult(vis=vis, time_s=time_s + done * TSAMP_S,
                              tint_samples=tint_samples, inputs=labels)
        done += used


def correlate(voltages, tint_s: float | None = None,
              tint_samples: int | None = None,
              inputs: list | None = None):
    """Visibilities from voltages: vis[t, f, i, j] = <v_i conj(v_j)>.

    Parameters
    ----------
    voltages : np.ndarray, dict, or iterable of gulps
        One of:

        * a (n_time, n_chan, n_input) complex array (what read_full_band
          gives with antenna_csv),
        * the raw snap dict {snap: (n_time, n_chan, n_adc)} — stacked along
          the input axis in (snap, adc) order, the pairs coming back in
          ``inputs``,
        * an iterable of either, or of FullBandResults — what
          ``VoltageReader.iter_full_band()`` yields. This form returns a
          generator instead of a single result.
    tint_s : float, optional
        Integration time in seconds, rounded to whole samples (one sample
        is 32.768 us). Default: no averaging.
    tint_samples : int, optional
        Integration length in samples; 1 (the default) keeps the native
        time resolution. Give this or tint_s, not both.
    inputs : list, optional
        Keep only these inputs, in this order: (snap, adc) pairs against
        the snap dict, column indices against an array. Default: every
        input read. The visibility cube and the work to build it both go
        as the square of the input count, so this is the lever that keeps
        a native-resolution correlation affordable. Selecting from the
        dict stacks only the wanted columns, so an unused ADC is never
        copied.

    Returns
    -------
    CorrelateResult, or a generator of them for a stream
        vis (n_bin, n_chan, n_input, n_input) complex64, time_s bin-centre
        times from the start of the array (or of the whole stream),
        tint_samples actually used, and the stacked (snap, adc) input
        labels for dict input. Samples past the last whole bin are dropped.

    A dump too long to hold in memory correlates the same way as a short
    one, the read handing over gulps as it goes:

    >>> vis_t, time_s = [], []
    >>> for out in correlate(reader.iter_full_band(snaps=[0, 3]),
    ...                      tint_s=0.001):
    ...     vis_t.append(out.vis.mean(axis=1))   # (n_bin, n_input, n_input)
    ...     time_s.append(out.time_s)
    >>> vis_t = np.concatenate(vis_t)

    The integration bins do not care where the gulps fall, so the result
    matches a single whole-dump call sample for sample.

    At native resolution pick the inputs you want, or the cube is 14.2 MB
    per sample for 24 inputs across the full band:

    >>> for out in correlate(reader.iter_full_band(snaps=[0, 3]),
    ...                      inputs=[(0, 0), (0, 1), (3, 0)]):
    ...     out.vis          # (n_bin, n_chan, 3, 3)
    ...     out.inputs       # [(0, 0), (0, 1), (3, 0)]
    """
    tint_samples = _resolve_tint(tint_s, tint_samples)

    if not isinstance(voltages, (np.ndarray, dict)) and \
            not hasattr(voltages, "voltages") and hasattr(voltages, "__iter__"):
        return _correlate_stream(voltages, tint_samples, inputs)

    v, labels = _as_inputs_array(voltages, inputs)

    n_bin = v.shape[0] // tint_samples
    if n_bin == 0:
        raise ValueError(
            f"integration of {tint_samples} samples is longer than the "
            f"{v.shape[0]} samples given"
        )

    vis = _correlate_block(v, tint_samples)
    time_s = (np.arange(n_bin) + 0.5) * tint_samples * TSAMP_S
    return CorrelateResult(vis=vis, time_s=time_s,
                           tint_samples=tint_samples, inputs=labels)
