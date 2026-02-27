"""
Correlator visibility file reader.

Reads raw .dat visibility files from the CASM correlator.
Files are named like: {base_str}.{index} where index is 0, 1, 2, ...
Each file contains ntime_per_file integrations of nchan x n_baselines x 2 (re/im) int32s.
"""

import gc
import glob
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from . import baselines
from .formats import VisibilityFormat


def extract_file_index(path: str) -> int:
    """Extract numeric index from file path (e.g., 'file.123' -> 123)."""
    m = re.search(r"\.(\d+)$", path)
    return int(m.group(1)) if m else -1


def discover_files(data_dir: str, base_str: str) -> dict[int, str]:
    """
    Find all .{index} files for an observation.

    Parameters
    ----------
    data_dir : str
        Directory containing visibility files.
    base_str : str
        Observation identifier (UTC timestamp like '2026-01-27-20:38:33').

    Returns
    -------
    dict
        Mapping of file index to full path.
    """
    pattern = os.path.join(data_dir, f"{base_str}.*")
    files = glob.glob(pattern)
    idx_to_path = {}
    for f in files:
        idx = extract_file_index(f)
        if idx >= 0:
            idx_to_path[idx] = f
    return idx_to_path


def read_visibilities(
    data_dir: str,
    base_str: str,
    fmt: VisibilityFormat,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
    time_tz: str = "UTC",
    ref: int | None = None,
    targets: list[int] | None = None,
    freq_order: str = "descending",
    verbose: bool = True,
) -> dict:
    """
    Read visibility data from .dat files.

    Parameters
    ----------
    data_dir : str
        Directory containing visibility .dat files.
    base_str : str
        Observation identifier (UTC timestamp).
    fmt : VisibilityFormat
        Format configuration (from load_format).
    time_start : str or datetime, optional
        Start time for slicing. None = from beginning.
    time_end : str or datetime, optional
        End time for slicing. None = to end.
    time_tz : str
        Timezone for time_start/end (default 'UTC').
    ref : int, optional
        Reference input index for baseline extraction.
        None = return all baselines.
    targets : list of int, optional
        Target input indices. None with ref = all other inputs.
    freq_order : str
        'descending' (default, native highest-freq-first) or 'ascending'.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        vis : np.ndarray
            (T, F, n_baselines) or (T, F, n_targets) complex64.
        freq_mhz : np.ndarray
            Frequency axis in MHz.
        time_unix : np.ndarray
            Unix timestamps per integration.
        metadata : dict
            Format info, files used, etc.
    """
    tz_obj = ZoneInfo(time_tz) if time_tz != "UTC" else timezone.utc

    def parse_dt(spec):
        if spec is None:
            return None
        if isinstance(spec, str):
            dt = datetime.fromisoformat(spec)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz_obj)
            return dt
        return spec.replace(tzinfo=tz_obj) if spec.tzinfo is None else spec

    utc_start_dt = parse_dt(time_start)
    utc_end_dt = parse_dt(time_end)
    if utc_start_dt:
        utc_start_dt = utc_start_dt.astimezone(timezone.utc)
    if utc_end_dt:
        utc_end_dt = utc_end_dt.astimezone(timezone.utc)

    # Discover files
    idx_to_path = discover_files(data_dir, base_str)
    if not idx_to_path:
        raise RuntimeError(f"No files found for {base_str} in {data_dir}")
    if 0 not in idx_to_path:
        raise RuntimeError("Missing required first file (suffix .0)")

    available_idxs = sorted(idx_to_path.keys())
    max_idx = available_idxs[-1]

    # Parse base time
    t0_unix = (
        datetime.strptime(base_str, "%Y-%m-%d-%H:%M:%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

    file_dur_s = fmt.file_duration_s
    data_start_unix = t0_unix
    data_end_unix = t0_unix + (max_idx + 1) * file_dur_s

    if verbose:
        print(f"Data span: {datetime.fromtimestamp(data_start_unix, tz=timezone.utc).isoformat()} -> "
              f"{datetime.fromtimestamp(data_end_unix, tz=timezone.utc).isoformat()}")
        print(f"Files available: {len(available_idxs)} (indices 0-{max_idx})")

    # Determine time range
    utc_start_unix = utc_start_dt.timestamp() if utc_start_dt else data_start_unix
    utc_end_unix = utc_end_dt.timestamp() if utc_end_dt else data_end_unix

    if utc_start_unix < data_start_unix or utc_end_unix > data_end_unix:
        raise ValueError(
            f"Requested time range outside available data.\n"
            f"Requested: {utc_start_unix} -> {utc_end_unix}\n"
            f"Available: {data_start_unix} -> {data_end_unix}"
        )

    # Convert to integration indices
    delta_start = utc_start_unix - t0_unix
    delta_end = utc_end_unix - t0_unix

    k_start = int(np.ceil(delta_start / fmt.dt_raw_s))
    k_stop = int(np.ceil(delta_end / fmt.dt_raw_s))
    if k_stop <= k_start:
        raise ValueError("Requested window contains zero integrations")

    file_start_idx = k_start // fmt.ntime_per_file
    file_stop_idx_excl = (k_stop - 1) // fmt.ntime_per_file + 1
    needed_file_idxs = list(range(file_start_idx, file_stop_idx_excl))

    missing = [i for i in needed_file_idxs if i not in idx_to_path]
    if missing:
        raise RuntimeError(f"Missing required .dat files for indices: {missing}")

    if verbose:
        print(f"Reading integrations {k_start}-{k_stop} from files "
              f"{needed_file_idxs[0]}-{needed_file_idxs[-1]}")

    # Build frequency axis
    freq_mhz = fmt.get_frequency_axis(order=freq_order)

    # Baseline extraction setup
    nbaseline = fmt.n_baselines
    bl_idx = None
    bl_conj = None
    extract_specific = ref is not None

    if extract_specific:
        if targets is None:
            targets = [i for i in range(fmt.nsig) if i != ref]
        bl_idx, bl_conj = baselines.build_baseline_plan(ref, targets, fmt.nsig)
        n_output = len(targets)
    else:
        n_output = nbaseline

    # Read files
    vis_chunks = []
    time_chunks = []
    kept_files = []

    denom = fmt.nchan * nbaseline * 2  # int32 count per integration

    for file_idx in needed_file_idxs:
        fpath = idx_to_path[file_idx]

        k0 = file_idx * fmt.ntime_per_file
        k1 = (file_idx + 1) * fmt.ntime_per_file
        s0 = max(k_start, k0) - k0
        s1 = min(k_stop, k1) - k0
        if s1 <= s0:
            continue

        if verbose:
            print(f"  Reading file .{file_idx} integrations [{s0}:{s1}]")

        raw = np.fromfile(fpath, dtype=np.int32)
        if raw.size % denom != 0:
            raise ValueError(
                f"File {fpath} has {raw.size} int32s, not divisible by {denom}"
            )

        ntime = raw.size // denom
        if ntime == 0:
            raise ValueError(
                f"File {fpath} too small for even one integration"
            )
        if ntime != fmt.ntime_per_file and verbose:
            print(f"  {os.path.basename(fpath)}: ntime={ntime} "
                  f"(expected {fmt.ntime_per_file})")

        # Cap slice to actual integrations available
        s1 = min(s1, ntime)

        xcorrs = raw.reshape(ntime, fmt.nchan, nbaseline, 2)
        del raw
        xcorrs = xcorrs[s0:s1, :, :, :]

        if extract_specific:
            xsel = xcorrs[:, :, bl_idx, :]
            del xcorrs
            v = xsel[..., 0].astype(np.float32) + 1j * xsel[..., 1].astype(np.float32)
            del xsel
            if np.any(bl_conj):
                v[:, :, bl_conj] = np.conj(v[:, :, bl_conj])
        else:
            v = xcorrs[..., 0].astype(np.float32) + 1j * xcorrs[..., 1].astype(np.float32)
            del xcorrs

        # Apply frequency ordering
        if freq_order == "ascending":
            v = v[:, ::-1, :]

        # Build times
        t_file_start = t0_unix + file_idx * file_dur_s
        t_local = t_file_start + fmt.dt_raw_s * np.arange(
            ntime, dtype=np.float64
        )
        t_sel = t_local[s0:s1]

        vis_chunks.append(v.astype(np.complex64))
        time_chunks.append(t_sel)
        kept_files.append(os.path.basename(fpath))
        gc.collect()

    if not vis_chunks:
        raise RuntimeError("No data collected after slicing")

    vis = np.concatenate(vis_chunks, axis=0)
    time_unix = np.concatenate(time_chunks)
    del vis_chunks, time_chunks
    gc.collect()

    if verbose:
        print(f"Output shape: {vis.shape} ({vis.dtype})")
        print(f"Frequency: {freq_mhz[0]:.3f} -> {freq_mhz[-1]:.3f} MHz ({freq_order})")
        print(f"Time samples: {len(time_unix)}")

    metadata = dict(
        format_name=fmt.name,
        nsig=fmt.nsig,
        dt_raw_s=fmt.dt_raw_s,
        nchan=fmt.nchan,
        base_str=base_str,
        data_dir=data_dir,
        files=kept_files,
        freq_order=freq_order,
    )

    if extract_specific:
        metadata["ref"] = ref
        metadata["targets"] = targets
        metadata["baseline_convention"] = "V(ref,target) with conjugation when ref>target"

    return dict(
        vis=vis,
        freq_mhz=freq_mhz,
        time_unix=time_unix,
        metadata=metadata,
    )
