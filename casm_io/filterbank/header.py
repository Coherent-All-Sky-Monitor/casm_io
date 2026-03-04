"""
Standalone SIGPROC filterbank header parser and writer.

Used as fallback when sigpyproc is not available.
Supports reading and writing the binary SIGPROC header format.
"""

import struct
import warnings
from pathlib import Path

import numpy as np


# SIGPROC header keyword types: 'i'=int32, 'd'=float64, 's'=string, 'b'=int8
HEADER_KEYWORDS = {
    "telescope_id": "i", "machine_id": "i", "data_type": "i",
    "rawdatafile": "s", "source_name": "s", "barycentric": "i",
    "pulsarcentric": "i", "az_start": "d", "za_start": "d",
    "src_raj": "d", "src_dej": "d", "tstart": "d", "tsamp": "d",
    "nbits": "i", "nsamples": "i", "fch1": "d", "foff": "d",
    "fchannel": "d", "nchans": "i", "nifs": "i", "refdm": "d",
    "flux": "d", "period": "d", "nbeams": "i", "ibeam": "i",
    "hdrlen": "i", "pb": "d", "ecc": "d", "asini": "d",
    "orig_hdrlen": "i", "new_hdrlen": "i", "sampsize": "i",
    "bandwidth": "d", "fbottom": "d", "ftop": "d",
    "obs_date": "s", "obs_time": "s", "signed": "b", "accel": "d",
}


def _read_string(f) -> str:
    """Read a length-prefixed string from filterbank file."""
    strlen = struct.unpack("i", f.read(4))[0]
    if strlen < 0 or strlen > 65536:
        raise ValueError(
            f"Invalid SIGPROC header string length: {strlen} "
            f"(must be 0-65536). File may be corrupted."
        )
    raw = f.read(strlen)
    if len(raw) != strlen:
        raise ValueError(
            f"Truncated SIGPROC header: expected {strlen} bytes, got {len(raw)}"
        )
    return raw.decode("utf-8")


def _write_string(f, s: str):
    """Write a length-prefixed string."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("i", len(encoded)))
    f.write(encoded)


def read_sigproc_header(filepath: str) -> tuple[dict, int]:
    """
    Read SIGPROC filterbank header.

    Parameters
    ----------
    filepath : str
        Path to .fil file.

    Returns
    -------
    header : dict
        Header key-value pairs.
    header_size : int
        Size of header in bytes.
    """
    header = {}
    with open(filepath, "rb") as f:
        keyword = _read_string(f)
        if keyword != "HEADER_START":
            raise ValueError(f"Expected HEADER_START, got {keyword}")

        while True:
            keyword = _read_string(f)
            if keyword == "HEADER_END":
                break
            if keyword in HEADER_KEYWORDS:
                dtype = HEADER_KEYWORDS[keyword]
                if dtype == "i":
                    header[keyword] = struct.unpack("i", f.read(4))[0]
                elif dtype == "d":
                    header[keyword] = struct.unpack("d", f.read(8))[0]
                elif dtype == "s":
                    header[keyword] = _read_string(f)
                elif dtype == "b":
                    header[keyword] = struct.unpack("b", f.read(1))[0]
            else:
                warnings.warn(
                    f"Unknown SIGPROC keyword '{keyword}' — stopping header parse",
                    stacklevel=2,
                )
                break

        header_size = f.tell()
    return header, header_size


def write_sigproc_header(f, header: dict):
    """
    Write SIGPROC binary header to an open file.

    Parameters
    ----------
    f : file object
        Open binary file for writing.
    header : dict
        Header key-value pairs to write.
    """
    _write_string(f, "HEADER_START")
    for key, val in header.items():
        if key.startswith("_"):
            continue
        if key not in HEADER_KEYWORDS:
            continue
        dtype = HEADER_KEYWORDS[key]
        _write_string(f, key)
        if dtype == "i":
            f.write(struct.pack("i", int(val)))
        elif dtype == "d":
            f.write(struct.pack("d", float(val)))
        elif dtype == "s":
            _write_string(f, str(val))
        elif dtype == "b":
            f.write(struct.pack("b", int(val)))
    _write_string(f, "HEADER_END")


def get_frequency_axis(header: dict) -> np.ndarray:
    """Get frequency axis in MHz from filterbank header."""
    nchans = header.get("nchans", 1)
    fch1 = header.get("fch1", 0.0)
    foff = header.get("foff", 1.0)
    if "nchans" not in header:
        warnings.warn("nchans missing from header, defaulting to 1", stacklevel=2)
    if "foff" not in header:
        warnings.warn("foff missing from header, defaulting to 1.0", stacklevel=2)
    return fch1 + np.arange(nchans) * foff


def get_time_axis(header: dict, nsamples: int | None = None) -> np.ndarray:
    """Get time axis in seconds from start."""
    if nsamples is None:
        nsamples = header.get("_nsamples", header.get("nsamples", 1))
    tsamp = header.get("tsamp", 1.0)
    return np.arange(nsamples) * tsamp
