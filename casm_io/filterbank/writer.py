"""
Unified filterbank writer.

Uses sigpyproc as primary backend. Falls back to standalone writer.
Every return dict includes 'backend_used' for traceability.
"""

import numpy as np

from .header import write_sigproc_header


def _write_standalone(filepath: str, data: np.ndarray, header: dict, nbits: int) -> dict:
    """Standalone writer using struct + numpy."""
    # Set CASM defaults if not specified
    if "telescope_id" not in header:
        header["telescope_id"] = 20  # OVRO

    write_header = {k: v for k, v in header.items() if not k.startswith("_")}
    write_header["nbits"] = nbits

    with open(filepath, "wb") as f:
        write_sigproc_header(f, write_header)

        if nbits == 8:
            if header.get("signed", 0):
                data.astype(np.int8).tofile(f)
            else:
                data.astype(np.uint8).tofile(f)
        elif nbits == 16:
            data.astype(np.uint16).tofile(f)
        elif nbits == 32:
            data.astype(np.float32).tofile(f)
        else:
            raise ValueError(f"Unsupported nbits: {nbits}")

    return dict(filepath=filepath, backend_used="standalone")


def _write_sigpyproc(filepath: str, data: np.ndarray, header: dict, nbits: int) -> dict:
    """Write using sigpyproc."""
    from sigpyproc.header import Header

    # Set CASM defaults
    if "telescope_id" not in header:
        header["telescope_id"] = 20

    # Build sigpyproc Header
    clean_header = {k: v for k, v in header.items() if not k.startswith("_")}
    sp_header = Header(**clean_header)

    out = sp_header.prep_outfile(filepath, nbits=nbits)
    # sigpyproc expects (nchans, nsamples) — transpose from our (nsamples, nchans)
    out.cwrite(data.T)
    out.close()

    return dict(filepath=filepath, backend_used="sigpyproc")


def write_filterbank(
    filepath: str,
    data: np.ndarray,
    header: dict,
    nbits: int = 8,
) -> dict:
    """
    Write a filterbank (.fil) file.

    Tries sigpyproc first, falls back to standalone writer.

    Parameters
    ----------
    filepath : str
        Output file path.
    data : np.ndarray
        Data array (nsamples, nchans).
    header : dict
        Filterbank header parameters.
    nbits : int
        Bits per sample (8, 16, or 32). Default 8.

    Returns
    -------
    dict with keys:
        filepath : str
            Path to written file.
        backend_used : str
            'sigpyproc' or 'standalone'.
    """
    try:
        return _write_sigpyproc(filepath, data, header, nbits)
    except (ImportError, Exception):
        return _write_standalone(filepath, data, header, nbits)
