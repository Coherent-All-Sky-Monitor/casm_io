"""
DADA header parser for voltage dump files.

The DADA header is a 4096-byte ASCII block at the start of each file.
Not all header fields are trustworthy — see TRUSTED_FIELDS and UNTRUSTED_FIELDS.

When trust_header=False (default), untrusted fields return hardcoded defaults
based on known correct values from the Feb 26 2026 header audit (FINDINGS.md).
"""

import os

HEADER_SIZE = 4096

# Fields verified correct in the Feb 26 2026 audit
TRUSTED_FIELDS = {
    "UTC_START",       # Per-dump timestamp — the ONLY reliable timing field
    "TSAMP",           # 32.768 microseconds — confirmed via data divisibility
    "NCHAN",           # 1024 channels per subband
    "NBIT",            # 4 bits per complex component
    "NDIM",            # 2 (real + imaginary)
    "ENCODING",        # TWOSCOMPLEMENT
    "RESOLUTION",      # 135168 = 11 SNAPs × 1024 chans × 12 ADCs
    "HDR_SIZE",        # 4096 bytes
    "BW",              # -31.25 MHz (negative = descending freq in raw data)
    "UDP_NANT",        # 6 UDP packets per integration
    "STREAM_SUBBAND_ID",  # Subband ordering (0, 1, 2)
    "FREQ",            # Subband center frequency (informational)
}

# Fields known to be wrong or unreliable
UNTRUSTED_FIELDS = {
    "NANT",            # Reports 66 — wrong. Should be 11 (SNAPs) or 16 (antennas)
    "FILE_SIZE",       # Pre-allocated max — use os.path.getsize() instead
    "SOURCE",          # Placeholder "J0000-0000" — can't ID observation from header
    "NPOL",            # Reports 2 — wrong. Files are single-pol (Pol A only)
    "PICOSECONDS",     # Inconsistent (0 or 7680000). Sub-us precision irrelevant at 32.768 us tsamp
    "START_CHANNEL",   # Off by 1024 due to global PFB indexing. Use directory names instead.
    "END_CHANNEL",     # Same issue as START_CHANNEL
}

# Hardcoded defaults for untrusted fields
_DEFAULTS = {
    "NANT": "11",       # 11 SNAP slots in data
    "FILE_SIZE": None,  # Must use os.path.getsize()
    "SOURCE": "UNKNOWN",
    "NPOL": "1",        # Single-pol
    "PICOSECONDS": "0",
    "START_CHANNEL": None,
    "END_CHANNEL": None,
}


def parse_dada_header(filename: str) -> dict[str, str]:
    """
    Read and parse the 4096-byte ASCII DADA header.

    Parameters
    ----------
    filename : str
        Path to DADA file.

    Returns
    -------
    dict
        All header key-value pairs as strings.
    """
    with open(filename, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
    header_text = header_bytes.decode("ascii", errors="ignore")
    header = {}
    for line in header_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            header[parts[0]] = parts[1].strip()
    return header


def get_header_value(
    header: dict[str, str],
    key: str,
    trust_header: bool = False,
    filename: str | None = None,
) -> str | None:
    """
    Get a header value, respecting trust mode.

    Parameters
    ----------
    header : dict
        Parsed header from parse_dada_header().
    key : str
        Header key to look up.
    trust_header : bool
        If True, return header value for ALL fields.
        If False (default), return hardcoded defaults for untrusted fields.
    filename : str, optional
        File path — needed for FILE_SIZE to use os.path.getsize().

    Returns
    -------
    str or None
        The header value, or None if unavailable.
    """
    if trust_header:
        return header.get(key)

    if key in TRUSTED_FIELDS:
        return header.get(key)

    if key in UNTRUSTED_FIELDS:
        if key == "FILE_SIZE" and filename is not None:
            return str(os.path.getsize(filename))
        return _DEFAULTS.get(key)

    # Unknown key — return header value with no guarantee
    return header.get(key)
