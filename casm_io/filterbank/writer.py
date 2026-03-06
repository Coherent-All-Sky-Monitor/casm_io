"""
Unified filterbank writer.

Uses standalone writer by default. sigpyproc backend available via
``backend='sigpyproc'``.

Every return dict includes 'backend_used' for traceability.
"""

import warnings

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
    """Write using sigpyproc.

    Notes
    -----
    sigpyproc's ``cwrite`` calls ``arr.tofile()`` on whatever array it
    receives.  Our data is ``(nsamples, nchans)`` C-contiguous, which is
    already time-major — the byte order SIGPROC expects.  We flatten it
    with ``data.ravel()`` so ``cwrite`` writes the correct byte stream
    without any transpose.
    """
    from sigpyproc.header import Header
    from sigpyproc.io.sigproc import telescope_ids

    # Build a clean copy, excluding internal keys
    sp = {k: v for k, v in header.items() if not k.startswith("_")}

    # Translate SIGPROC-style keys -> sigpyproc Header kwargs
    # source_name -> source
    if "source_name" in sp and "source" not in sp:
        sp["source"] = sp.pop("source_name")
    elif "source_name" in sp:
        sp.pop("source_name")

    # telescope_id (int) -> telescope (str)
    if "telescope_id" in sp and "telescope" not in sp:
        tid = sp.pop("telescope_id")
        inv = {v: k for k, v in telescope_ids.items()}
        sp["telescope"] = inv.get(tid, "Fake")
    elif "telescope_id" in sp:
        sp.pop("telescope_id")

    # data_type: SIGPROC uses int (1=filterbank), sigpyproc uses str
    if "data_type" in sp and isinstance(sp["data_type"], int):
        sp["data_type"] = {1: "filterbank", 2: "timeseries"}.get(
            sp["data_type"], "filterbank"
        )

    # Ensure required defaults
    sp.setdefault("telescope", "Fake")
    sp.setdefault("data_type", "filterbank")
    sp.setdefault("filename", "")

    # Remove keys that sigpyproc Header doesn't accept
    import inspect
    valid_keys = set(inspect.signature(Header.__init__).parameters.keys()) - {"self"}
    sp = {k: v for k, v in sp.items() if k in valid_keys}

    # sigpyproc writes empty rawdatafile as a length-0 string which its own
    # reader then rejects as corrupt -- ensure it's always non-empty
    if not sp.get("rawdatafile"):
        sp["rawdatafile"] = filepath

    sp_header = Header(**sp)

    out = sp_header.prep_outfile(filepath, nbits=nbits)
    # Feed data as a flat, C-contiguous (time-major) array so cwrite's
    # tofile() produces the correct SIGPROC byte order.
    out.cwrite(data.ravel())
    out.close()

    return dict(filepath=filepath, backend_used="sigpyproc")


def write_filterbank(
    filepath: str,
    data: np.ndarray,
    header: dict,
    nbits: int = 8,
    backend: str = "standalone",
) -> dict:
    """
    Write a filterbank (.fil) file.

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
    backend : str
        Writer backend: ``'standalone'`` (default) or ``'sigpyproc'``.

    Returns
    -------
    dict with keys:
        filepath : str
            Path to written file.
        backend_used : str
            'sigpyproc' or 'standalone'.
    """
    if backend == "sigpyproc":
        try:
            return _write_sigpyproc(filepath, data, header, nbits)
        except ImportError:
            warnings.warn(
                "sigpyproc not available, falling back to standalone writer",
                stacklevel=2,
            )
        except (TypeError, AttributeError) as e:
            warnings.warn(
                f"sigpyproc write failed ({e}), falling back to standalone writer",
                stacklevel=2,
            )
    return _write_standalone(filepath, data, header, nbits)
