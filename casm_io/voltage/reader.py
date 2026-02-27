"""
Voltage dump DADA file reader.

Supports:
- Single subband files (1024 channels)
- Full 3-subband dumps (3072 channels, stitched)
- Optional per-antenna extraction via antenna mapping CSV
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .header import parse_dada_header, HEADER_SIZE
from .unpack import unpack_4bit

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_dada_config() -> dict:
    """Load default DADA format config."""
    config_path = os.path.join(_CONFIGS_DIR, "dada_format.json")
    with open(config_path) as f:
        return json.load(f)


def _make_freq_axis_subband(
    subband_index: int, n_chan: int, freq_top_mhz: float,
    chan_bw_mhz: float, n_subbands: int, order: str,
) -> np.ndarray:
    """Build frequency axis for one subband."""
    # Native order: descending from freq_top.
    # Subband 0 = highest freqs, subband 2 = lowest freqs.
    n_chan_total = n_chan * n_subbands
    start_chan = subband_index * n_chan
    freqs_desc = (
        freq_top_mhz
        - chan_bw_mhz * np.arange(start_chan, start_chan + n_chan, dtype=np.float64)
    )
    if order == "ascending":
        return freqs_desc[::-1].copy()
    return freqs_desc


def read_dada_file(
    filename: str,
    n_time: int | None = None,
    snaps: list[int] | None = None,
    freq_order: str = "descending",
    trust_header: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Read a single DADA subband file.

    Parameters
    ----------
    filename : str
        Path to .dada file.
    n_time : int, optional
        Number of time samples to read. None = all.
    snaps : list of int, optional
        SNAP slot indices to extract. Default: active_snaps from config.
    freq_order : str
        'descending' (default, native) or 'ascending'.
    trust_header : bool
        If True, trust all header fields. Default False.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        voltages : dict
            {snap_id: (n_time, n_chan_per_sub, n_adc) complex64}
        header : dict
            Parsed DADA header.
        freq_mhz : np.ndarray
            Frequency axis for this subband.
    """
    cfg = _load_dada_config()
    n_snaps = cfg["n_snap_slots"]
    n_adc = cfg["n_adc_per_snap"]
    n_chan_sub = cfg["n_chan_per_subband"]

    if snaps is None:
        snaps = cfg["active_snaps"]

    header = parse_dada_header(filename)
    file_size = os.path.getsize(filename)
    data_size = file_size - HEADER_SIZE
    bytes_per_time = n_snaps * n_chan_sub * n_adc
    n_time_total = data_size // bytes_per_time

    if n_time is None:
        n_time = n_time_total
    else:
        n_time = min(n_time, n_time_total)

    if verbose:
        print(f"Reading {filename}")
        print(f"  n_time={n_time} / {n_time_total}, SNAPs={snaps}")

    n_bytes = n_time * bytes_per_time
    with open(filename, "rb") as f:
        f.seek(HEADER_SIZE)
        raw = np.frombuffer(f.read(n_bytes), dtype=np.uint8)

    raw = raw.reshape(n_time, n_snaps, n_chan_sub, n_adc)

    voltages = {}
    for s in snaps:
        voltages[s] = unpack_4bit(raw[:, s, :, :])
        if freq_order == "ascending":
            voltages[s] = voltages[s][:, ::-1, :]
        if verbose:
            p = np.mean(np.abs(voltages[s]) ** 2)
            print(f"  SNAP {s}: mean_power={p:.3f}")

    # Determine subband index from filename or directory
    subband_index = _guess_subband_index(filename)

    freq_mhz = _make_freq_axis_subband(
        subband_index, n_chan_sub, cfg["freq_top_mhz"],
        cfg["chan_bw_mhz"], cfg["n_subbands"], freq_order,
    )

    return dict(voltages=voltages, header=header, freq_mhz=freq_mhz)


def _guess_subband_index(filename: str) -> int:
    """Guess subband index (0=highest freq, 2=lowest) from filename/path."""
    path_lower = filename.lower()
    if "chan0_1023" in path_lower:
        return 0
    elif "chan1024_2047" in path_lower:
        return 1
    elif "chan2048_3071" in path_lower:
        return 2
    return 0  # default


def read_voltage_dump(
    data_dir: str,
    timestamp: str,
    antenna_csv: str | None = None,
    n_time: int | None = None,
    snaps: list[int] | None = None,
    freq_order: str = "descending",
    trust_header: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Read a full 3-subband voltage dump and stitch into full band.

    Parameters
    ----------
    data_dir : str
        Base directory containing subband subdirectories
        (chan0_1023/, chan1024_2047/, chan2048_3071/).
    timestamp : str
        UTC timestamp string (e.g. '2026-02-17-21:10:43').
    antenna_csv : str, optional
        Path to antenna mapping CSV. If provided, extracts per-antenna
        voltages as (n_time, 3072, n_ant). If not, returns raw per-SNAP dicts.
    n_time : int, optional
        Time samples to read per subband. None = all.
    snaps : list of int, optional
        SNAP indices to extract. Default: active_snaps from config.
    freq_order : str
        'descending' (default, native) or 'ascending'.
    trust_header : bool
        Trust all DADA header fields. Default False.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        voltages : np.ndarray or dict
            If antenna_csv: (n_time, 3072, n_ant) complex64.
            If not: {snap_id: (n_time, 3072, n_adc) complex64}.
        header : dict
            Header from first subband file.
        freq_mhz : np.ndarray
            Full 3072-channel frequency axis.
        utc_start : str
            UTC_START from header.
        antenna_df : pd.DataFrame or None
            Antenna mapping if csv was provided.
    """
    cfg = _load_dada_config()
    subband_dirs = cfg["subband_dirs"]

    if snaps is None:
        snaps = cfg["active_snaps"]

    # Find subband files
    sub_files = []
    for sub_dir in subband_dirs:
        sub_path = os.path.join(data_dir, sub_dir)
        if not os.path.isdir(sub_path):
            raise FileNotFoundError(
                f"Subband directory not found: {sub_path}"
            )
        # Find the file matching this timestamp
        matches = [
            f for f in os.listdir(sub_path)
            if f.startswith(timestamp) and f.endswith(".dada")
        ]
        if not matches:
            raise FileNotFoundError(
                f"No DADA file matching timestamp '{timestamp}' in {sub_path}"
            )
        sub_files.append(os.path.join(sub_path, sorted(matches)[0]))

    # Read each subband
    all_voltages = {s: [] for s in snaps}
    first_header = None
    utc_start = "unknown"

    for i, sf in enumerate(sub_files):
        if verbose:
            print(f"\n--- Subband {i} ({subband_dirs[i]}) ---")
        result = read_dada_file(
            sf, n_time=n_time, snaps=snaps,
            freq_order=freq_order, trust_header=trust_header,
            verbose=verbose,
        )
        if first_header is None:
            first_header = result["header"]
            utc_start = first_header.get("UTC_START", "unknown")
        for s in snaps:
            all_voltages[s].append(result["voltages"][s])

    # Stitch along frequency axis
    # Native order: sub 0 = highest freqs, sub 2 = lowest
    # When ascending: we reverse each subband AND reverse subband order
    voltages_stitched = {}
    for s in snaps:
        if freq_order == "ascending":
            # Subbands already individually reversed in read_dada_file
            # But subband ORDER needs reversing: sub2 (lowest) first
            voltages_stitched[s] = np.concatenate(
                all_voltages[s][::-1], axis=1
            )
        else:
            # Descending: sub0 (highest) first — already correct order
            voltages_stitched[s] = np.concatenate(all_voltages[s], axis=1)

        if verbose:
            print(f"\nSNAP {s} stitched: shape={voltages_stitched[s].shape}, "
                  f"mean_power={np.mean(np.abs(voltages_stitched[s]) ** 2):.3f}")

    # Build full frequency axis
    n_chan_total = cfg["n_chan_total"]
    freq_top = cfg["freq_top_mhz"]
    chan_bw = cfg["chan_bw_mhz"]
    freq_desc = freq_top - chan_bw * np.arange(n_chan_total, dtype=np.float64)
    if freq_order == "ascending":
        freq_mhz = freq_desc[::-1].copy()
    else:
        freq_mhz = freq_desc

    if verbose:
        print(f"\nFrequency axis: {freq_mhz[0]:.3f} – {freq_mhz[-1]:.3f} MHz "
              f"({n_chan_total} channels, {freq_order})")
        print(f"UTC_START: {utc_start}")

    # Extract per-antenna voltages if CSV provided
    antenna_df = None
    if antenna_csv is not None:
        antenna_df = pd.read_csv(antenna_csv)
        # Support legacy column names
        rename = {}
        if "antenna" in antenna_df.columns and "antenna_id" not in antenna_df.columns:
            rename["antenna"] = "antenna_id"
        if "snap" in antenna_df.columns and "snap_id" not in antenna_df.columns:
            rename["snap"] = "snap_id"
        if rename:
            antenna_df = antenna_df.rename(columns=rename)

        n_ant = len(antenna_df)
        s0 = list(voltages_stitched.keys())[0]
        nt, nc, _ = voltages_stitched[s0].shape

        ant_voltages = np.zeros((nt, nc, n_ant), dtype=np.complex64)
        for i, (_, row) in enumerate(antenna_df.iterrows()):
            snap = int(row["snap_id"])
            adc = int(row["adc"])
            if snap in voltages_stitched:
                ant_voltages[:, :, i] = voltages_stitched[snap][:, :, adc]
                if verbose:
                    p = np.mean(np.abs(ant_voltages[:, :, i]) ** 2)
                    print(f"  Ant {int(row['antenna_id']):2d}: "
                          f"SNAP {snap} ADC {adc:2d} → power={p:.3f}")
            else:
                if verbose:
                    print(f"  Ant {int(row['antenna_id']):2d}: "
                          f"SNAP {snap} NOT LOADED — zeros")

        return dict(
            voltages=ant_voltages,
            header=first_header,
            freq_mhz=freq_mhz,
            utc_start=utc_start,
            antenna_df=antenna_df,
        )

    return dict(
        voltages=voltages_stitched,
        header=first_header,
        freq_mhz=freq_mhz,
        utc_start=utc_start,
        antenna_df=None,
    )
