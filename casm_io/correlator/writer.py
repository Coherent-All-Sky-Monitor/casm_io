"""
Correlator visibility NPZ writer and reader.
"""

from datetime import datetime, timezone

import numpy as np

from .._results import VisibilityResult
from .formats import VisibilityFormat


def write_npz(
    path: str,
    vis: np.ndarray,
    freq_mhz: np.ndarray,
    time_unix: np.ndarray,
    ref: int | None = None,
    targets: np.ndarray | list | None = None,
    fmt: VisibilityFormat | None = None,
    **metadata,
):
    """
    Write visibility data to NPZ with metadata.

    Parameters
    ----------
    path : str
        Output file path.
    vis : np.ndarray
        Visibility array (T, F, nbaseline) or (T, F, ntarget).
    freq_mhz : np.ndarray
        Frequency axis in MHz.
    time_unix : np.ndarray
        Unix timestamps.
    ref : int, optional
        Reference input index.
    targets : array-like, optional
        Target input indices.
    fmt : VisibilityFormat, optional
        Format used.
    **metadata
        Additional metadata.
    """
    time_utc_iso = np.array(
        [
            datetime.fromtimestamp(t, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            )
            + "Z"
            for t in time_unix
        ],
        dtype="U30",
    )

    save_dict = dict(
        vis=vis,
        freq_mhz=freq_mhz.astype(np.float32),
        time_utc_unix=time_unix.astype(np.float64),
        time_utc_iso=time_utc_iso,
    )

    if ref is not None:
        save_dict["ref_input"] = np.int64(ref)
    if targets is not None:
        save_dict["target_inputs"] = np.array(targets, dtype=np.int64)

    if fmt is not None:
        save_dict["format_name"] = fmt.name
        save_dict["dt_raw_s"] = np.float64(fmt.dt_raw_s)
        save_dict["chan_bw_mhz"] = np.float64(fmt.chan_bw_mhz)

    if ref is not None:
        save_dict["baseline_convention"] = (
            "V(ref,target) with Hermitian conj when ref>target"
        )

    for k, v in metadata.items():
        if isinstance(v, str):
            save_dict[k] = v
        elif isinstance(v, (int, float)):
            save_dict[k] = np.array(v)
        elif isinstance(v, (list, tuple)):
            save_dict[k] = np.array(v, dtype=object)
        else:
            save_dict[k] = v

    np.savez(path, **save_dict)


def read_npz(path: str) -> dict:
    """
    Read visibility NPZ file.

    Returns
    -------
    dict with keys: vis, freq_mhz, time_unix, ref, targets, metadata
    """
    D = np.load(path, allow_pickle=True)

    if "vis" in D:
        vis = D["vis"]
    elif "vis_ref" in D:
        vis = D["vis_ref"]
    else:
        raise ValueError("NPZ missing 'vis' or 'vis_ref' key")

    ref = int(D["ref_input"]) if "ref_input" in D else (
        int(D["ref_adc"]) if "ref_adc" in D else None
    )
    targets = (
        D["target_inputs"].astype(np.int64)
        if "target_inputs" in D
        else (D["target_adcs"].astype(np.int64) if "target_adcs" in D else None)
    )

    skip_keys = {
        "vis", "vis_ref", "freq_mhz", "time_utc_unix", "time_unix",
        "ref_input", "ref_adc", "target_inputs", "target_adcs",
    }
    metadata = {}
    for key in D.files:
        if key not in skip_keys:
            try:
                val = D[key]
                metadata[key] = val.item() if val.ndim == 0 else val
            except Exception:
                metadata[key] = D[key]

    return VisibilityResult(
        vis=vis.astype(np.complex64),
        freq_mhz=D["freq_mhz"].astype(np.float64),
        time_unix=(
            D["time_utc_unix"].astype(np.float64)
            if "time_utc_unix" in D
            else D["time_unix"].astype(np.float64)
        ),
        metadata=metadata,
        ref=ref,
        targets=targets,
    )
