"""
Unified filterbank reader.

Uses sigpyproc as primary backend (fast C-backed reading).
Falls back to standalone struct-based reader if sigpyproc unavailable.

Every return dict includes 'backend_used' for traceability.
"""

from pathlib import Path

import numpy as np

from .header import read_sigproc_header, get_frequency_axis, get_time_axis


def _read_standalone(filepath: str) -> dict:
    """Standalone reader using struct + numpy."""
    header, header_size = read_sigproc_header(filepath)

    nchans = header.get("nchans", 1)
    nbits = header.get("nbits", 8)
    nifs = header.get("nifs", 1)

    bytes_per_sample = nchans * nifs * nbits // 8
    file_size = Path(filepath).stat().st_size
    data_size = file_size - header_size
    nsamples = data_size // bytes_per_sample

    with open(filepath, "rb") as f:
        f.seek(header_size)
        if nbits == 8:
            dtype = np.int8 if header.get("signed", 0) else np.uint8
            data = np.fromfile(f, dtype=dtype, count=nsamples * nchans * nifs)
        elif nbits == 16:
            data = np.fromfile(f, dtype=np.uint16, count=nsamples * nchans * nifs)
        elif nbits == 32:
            data = np.fromfile(f, dtype=np.float32, count=nsamples * nchans * nifs)
        else:
            raise ValueError(f"Unsupported nbits: {nbits}")

    if nifs > 1:
        data = data.reshape(nsamples, nifs, nchans)[:, 0, :]
    else:
        data = data.reshape(nsamples, nchans)

    header["_nsamples"] = nsamples
    header["_header_size"] = header_size

    return dict(
        header=header,
        data=data,
        freq_mhz=get_frequency_axis(header),
        time_s=get_time_axis(header, nsamples),
        backend_used="standalone",
    )


def _read_sigpyproc(filepath: str) -> dict:
    """Read using sigpyproc."""
    from sigpyproc.readers import FilReader

    fil = FilReader(filepath)
    block = fil.read_block(0, fil.header.nsamples)

    # sigpyproc returns FilterbankBlock with .data of shape (nchans, nsamples)
    data = block.data.T

    header = fil.header.to_dict()
    header["_nsamples"] = fil.header.nsamples
    nchans = header.get("nchans", data.shape[1])
    fch1 = header.get("fch1", 0.0)
    foff = header.get("foff", 1.0)
    freq_mhz = fch1 + np.arange(nchans) * foff
    tsamp = header.get("tsamp", 1.0)
    time_s = np.arange(data.shape[0]) * tsamp

    return dict(
        header=header,
        data=data,
        freq_mhz=freq_mhz,
        time_s=time_s,
        backend_used="sigpyproc",
    )


def read_filterbank(filepath: str) -> dict:
    """
    Read a filterbank (.fil) file.

    Tries sigpyproc first, falls back to standalone reader.

    Parameters
    ----------
    filepath : str
        Path to .fil file.

    Returns
    -------
    dict with keys:
        header : dict
            Filterbank header parameters.
        data : np.ndarray
            Data array (nsamples, nchans).
        freq_mhz : np.ndarray
            Frequency axis in MHz.
        time_s : np.ndarray
            Time axis in seconds from start.
        backend_used : str
            'sigpyproc' or 'standalone'.
    """
    try:
        return _read_sigpyproc(filepath)
    except ImportError:
        return _read_standalone(filepath)
