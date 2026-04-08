"""
Correlator visibility file header parser.

Starting March 4, 2026, correlator .dat files have a 4096-byte ASCII header.
Old files have no header. Detection is automatic via 'HDR_SIZE' substring check.
"""

import math

from .formats import VisibilityFormat

HEADER_SIZE = 4096


def has_header(filepath: str) -> bool:
    """Check if a visibility file has a 4096-byte ASCII header."""
    with open(filepath, "rb") as f:
        raw = f.read(HEADER_SIZE)
    text = raw.decode("ascii", errors="ignore")
    return "HDR_SIZE" in text


def parse_corr_header(filepath: str) -> dict[str, str]:
    """
    Read and parse the 4096-byte ASCII header from a visibility file.

    Parameters
    ----------
    filepath : str
        Path to a .dat visibility file with header.

    Returns
    -------
    dict
        Header key-value pairs as strings.
    """
    with open(filepath, "rb") as f:
        raw = f.read(HEADER_SIZE)
    text = raw.decode("ascii", errors="ignore")
    header = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            header[parts[0]] = parts[1].strip()
    return header


def get_header_offset(filepath: str) -> tuple[int, dict | None]:
    """
    Detect header presence and return offset + parsed header.

    Returns
    -------
    tuple of (int, dict or None)
        (4096, header_dict) if header present, (0, None) if not.
    """
    if has_header(filepath):
        return HEADER_SIZE, parse_corr_header(filepath)
    return 0, None


def format_from_header(header: dict[str, str]) -> VisibilityFormat:
    """
    Build a VisibilityFormat from parsed header fields.

    Header fields used:
        NCHAN, NBASELINE, CORR_DUMP_DUMPS_PER_FILE, TSAMP (picoseconds),
        FREQ_START, CHANBW (negative = descending)

    Parameters
    ----------
    header : dict
        Parsed header from parse_corr_header().

    Returns
    -------
    VisibilityFormat
    """
    nchan = int(header["NCHAN"])
    # NBASELINE in newer headers, NBEAM in early headers (same value)
    if "NBASELINE" in header:
        n_baselines = int(header["NBASELINE"])
    elif "NBEAM" in header:
        n_baselines = int(header["NBEAM"])
    else:
        raise KeyError("Header missing both NBASELINE and NBEAM")
    ntime_per_file = int(header["CORR_DUMP_DUMPS_PER_FILE"])
    dt_raw_s = float(header["TSAMP"]) / 1e6  # microseconds -> seconds
    # FREQ_START in newer headers; fall back to FREQ + BW/2 for early headers
    if "FREQ_START" in header:
        freq_top_mhz = float(header["FREQ_START"])
    elif "FREQ" in header and "BW" in header:
        freq_top_mhz = float(header["FREQ"]) + abs(float(header["BW"])) / 2
    else:
        raise KeyError("Header missing FREQ_START (and no FREQ+BW fallback)")
    # CHANBW in newer headers; derive from BW/NCHAN for early headers
    if "CHANBW" in header:
        chanbw_raw = float(header["CHANBW"])
    elif "BW" in header:
        chanbw_raw = float(header["BW"]) / nchan
    else:
        raise KeyError("Header missing both CHANBW and BW")
    native_order = "descending" if chanbw_raw < 0 else "ascending"
    chan_bw_mhz = abs(chanbw_raw)

    # Derive nsig from n_baselines: n_baselines = nsig*(nsig+1)/2
    # nsig = (-1 + sqrt(1 + 8*n_baselines)) / 2
    nsig = int((-1 + math.sqrt(1 + 8 * n_baselines)) / 2)
    if nsig * (nsig + 1) // 2 != n_baselines:
        raise ValueError(
            f"NBASELINE={n_baselines} does not correspond to a valid nsig"
        )

    freq_bottom_mhz = freq_top_mhz - nchan * chan_bw_mhz

    return VisibilityFormat(
        name="from_header",
        nsig=nsig,
        dt_raw_s=dt_raw_s,
        ntime_per_file=ntime_per_file,
        nchan=nchan,
        chan_bw_mhz=chan_bw_mhz,
        freq_top_mhz=freq_top_mhz,
        freq_bottom_mhz=freq_bottom_mhz,
        native_order=native_order,
    )
