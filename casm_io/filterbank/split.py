"""
Filterbank time-slice extraction.

Extracts a range of samples from a filterbank file and writes a new
filterbank with an adjusted header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .header import read_sigproc_header
from .writer import write_filterbank


def split_filterbank(
    input_path: str,
    output_path: str,
    start_sample: int = 0,
    nsamples: int | None = None,
    beam: int | None = None,
    verbose: bool = True,
) -> dict:
    """Extract a time range from a filterbank file.

    Reads samples ``[start_sample, start_sample + nsamples)`` from the
    input file, adjusts the header (``tstart`` shifted forward, ``nsamples``
    updated), and writes a new single-beam filterbank file.

    Uses seek + fromfile to avoid loading the entire file into memory.

    Parameters
    ----------
    input_path : str
        Path to input filterbank file.
    output_path : str
        Path for output filterbank file.
    start_sample : int
        First sample to extract (0-indexed). Default 0.
    nsamples : int or None
        Number of samples to extract. ``None`` means read to end of file.
    beam : int or None
        Beam index to extract from a multibeam file. Ignored for
        single-beam files. Defaults to 0 for multibeam.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Keys: ``filepath``, ``nsamples_written``, ``header``.

    Raises
    ------
    ValueError
        If start_sample or nsamples are out of range.
    """
    header, header_size = read_sigproc_header(input_path)

    nchans = header.get("nchans", 1)
    nbits = header.get("nbits", 8)
    nbeams = max(header.get("nbeams", 1), 1)
    tsamp = header.get("tsamp", 1.0)

    if nbits < 8:
        raise ValueError(
            f"Sub-byte filterbanks (nbits={nbits}) not supported by split_filterbank"
        )

    # Compute total samples from file size
    file_size = Path(input_path).stat().st_size
    data_bytes = file_size - header_size
    bytes_per_value = nbits // 8
    total_samples_all_beams = data_bytes // (nchans * bytes_per_value)
    samples_per_beam = total_samples_all_beams // nbeams

    # Resolve beam
    if nbeams > 1:
        if beam is None:
            beam = 0
        if beam < 0 or beam >= nbeams:
            raise ValueError(
                f"beam={beam} out of range for {nbeams}-beam file"
            )
    else:
        beam = 0

    # Validate range
    if start_sample < 0 or start_sample >= samples_per_beam:
        raise ValueError(
            f"start_sample={start_sample} out of range "
            f"(file has {samples_per_beam} samples per beam)"
        )

    max_available = samples_per_beam - start_sample
    if nsamples is None:
        nsamples = max_available
    elif nsamples > max_available:
        raise ValueError(
            f"Requested {nsamples} samples from offset {start_sample}, "
            f"but only {max_available} available"
        )

    if verbose:
        print(f"Splitting {input_path}")
        print(f"  Extracting samples [{start_sample}, {start_sample + nsamples})")
        print(f"  nchans={nchans}, nbits={nbits}, nbeams={nbeams}, beam={beam}")

    # Determine dtype
    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.float32}
    if nbits == 8 and header.get("signed", 0):
        dtype = np.int8
    else:
        dtype = dtype_map.get(nbits)
        if dtype is None:
            raise ValueError(f"Unsupported nbits: {nbits}")

    # Compute byte offset into the data section
    beam_stride = samples_per_beam * nchans * bytes_per_value
    sample_stride = nchans * bytes_per_value
    data_offset = header_size + beam * beam_stride + start_sample * sample_stride

    # Read the slice
    count = nsamples * nchans
    with open(input_path, "rb") as f:
        f.seek(data_offset)
        data = np.fromfile(f, dtype=dtype, count=count)

    if data.size != count:
        raise ValueError(
            f"Truncated read: expected {count} values, got {data.size}"
        )

    data = data.reshape(nsamples, nchans)

    # Build output header
    out_header = {k: v for k, v in header.items() if not k.startswith("_")}
    out_header["tstart"] = header.get("tstart", 0.0) + start_sample * tsamp / 86400.0
    out_header["nsamples"] = nsamples
    out_header["nbeams"] = 1
    out_header.pop("ibeam", None)

    # Remove keys that the writer doesn't need
    skip_keys = {"az_start", "za_start", "rawdatafile"}
    out_header = {k: v for k, v in out_header.items() if k not in skip_keys}

    # Write output
    result = write_filterbank(
        filepath=output_path,
        data=data,
        header=out_header,
        nbits=nbits,
    )

    if verbose:
        duration_s = nsamples * tsamp
        print(f"  Written {nsamples} samples ({duration_s:.2f} s) to {output_path}")

    return {
        "filepath": output_path,
        "nsamples_written": nsamples,
        "header": out_header,
    }
