"""
Filterbank requantization.

Converts filterbank files between bit depths (e.g. 32-bit float to
8-bit unsigned int) using per-channel linear scaling that preserves
the natural bandpass shape and noise structure.
"""

from __future__ import annotations

import numpy as np

from .header import read_sigproc_header
from .reader import FilterbankFile
from .writer import write_filterbank


def requantize_filterbank(
    input_path: str,
    output_path: str,
    target_nbits: int = 8,
    sigma_clip: float = 4.0,
    beam: int | None = None,
    verbose: bool = True,
) -> dict:
    """Requantize a filterbank file to a different bit depth.

    Performs per-channel linear mapping so that each channel's median
    maps to the mid-range value (e.g. 128 for uint8) and
    ``+/- sigma_clip`` standard deviations span the full output range.
    This preserves the natural bandpass shape — channels with higher
    power keep higher mean values in the output.

    Parameters
    ----------
    input_path : str
        Path to input filterbank file.
    output_path : str
        Path for output filterbank file.
    target_nbits : int
        Target bit depth (8 or 16). Default 8.
    sigma_clip : float
        Number of standard deviations mapped to half the output range.
        Default 4.0.
    beam : int or None
        Beam index for multibeam files.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    dict
        Keys: ``filepath``, ``nsamples``, ``nchans``,
        ``per_channel_stats`` (dict with median/std/scale/offset arrays),
        ``backend_used``.
    """
    if target_nbits not in (8, 16):
        raise ValueError(f"target_nbits must be 8 or 16, got {target_nbits}")

    fb = FilterbankFile(input_path, beam=beam, verbose=verbose)
    data = fb.data  # (nsamples, nchans), any dtype

    max_val = (1 << target_nbits) - 1  # 255 for 8-bit
    mid_val = max_val / 2.0            # 127.5 for 8-bit

    nchans = data.shape[1]
    data_f64 = data.astype(np.float64)
    chan_median = np.median(data_f64, axis=0)
    chan_std = np.std(data_f64, axis=0)

    # Per-channel scale: maps sigma_clip * std to half the output range
    # output = (input - median) * scale + mid_val
    scale = np.zeros(nchans, dtype=np.float64)
    alive = chan_std > 0
    scale[alive] = mid_val / (sigma_clip * chan_std[alive])

    if verbose:
        n_dead = int(np.sum(~alive))
        print(f"Requantizing {input_path} -> {target_nbits}-bit")
        print(f"  {nchans} channels, {n_dead} dead (zero std)")
        print(f"  sigma_clip={sigma_clip}, mid_val={mid_val}")
        if np.any(alive):
            print(f"  Input median range: {chan_median[alive].min():.4f} - "
                  f"{chan_median[alive].max():.4f}")
            print(f"  Input std range: {chan_std[alive].min():.6f} - "
                  f"{chan_std[alive].max():.6f}")
            print(f"  Scale range: {scale[alive].min():.2f} - "
                  f"{scale[alive].max():.2f}")

    # Apply per-channel linear mapping (reuse data_f64 from stats computation)
    requantized = (data_f64 - chan_median[np.newaxis, :]) * scale[np.newaxis, :] + mid_val
    del data_f64  # free memory before allocating output

    # Clip and cast
    if target_nbits == 8:
        out_data = np.clip(np.round(requantized), 0, max_val).astype(np.uint8)
    else:
        out_data = np.clip(np.round(requantized), 0, max_val).astype(np.uint16)

    # Dead channels -> 0
    out_data[:, ~alive] = 0

    if verbose and np.any(alive):
        print(f"  Output mean range: {np.mean(out_data[:, alive].astype(float), axis=0).min():.1f} - "
              f"{np.mean(out_data[:, alive].astype(float), axis=0).max():.1f}")
        print(f"  Output std range: {np.std(out_data[:, alive].astype(float), axis=0).min():.2f} - "
              f"{np.std(out_data[:, alive].astype(float), axis=0).max():.2f}")

    # Use raw SIGPROC header to avoid sigpyproc-added keys (accel,
    # bandwidth, fbottom, ftop, obs_time, period, etc.) that cause
    # round-trip failures with sigpyproc's strict header parser.
    raw_header, _ = read_sigproc_header(input_path)
    out_header = {k: v for k, v in raw_header.items() if not k.startswith("_")}
    out_header["nbits"] = target_nbits

    result = write_filterbank(
        filepath=output_path,
        data=out_data,
        header=out_header,
        nbits=target_nbits,
    )

    if verbose:
        print(f"  Written to {output_path}")

    return {
        "filepath": output_path,
        "nsamples": fb.nsamples,
        "nchans": nchans,
        "per_channel_stats": {
            "median": chan_median,
            "std": chan_std,
            "scale": scale,
        },
        "backend_used": result.get("backend_used", "standalone"),
    }
