"""
casm_io.correlator — Correlator visibility file I/O.

Two formats supported:
- layout_32ant: 64 inputs (32 ant x 2 pol), 34.36s integrations
- layout_64ant: 128 inputs (64 ant x 2 pol), 137.44s integrations
"""

from .formats import load_format, VisibilityFormat
from .header import has_header, parse_corr_header, get_header_offset, format_from_header
from .reader import VisibilityReader, discover_files
from .writer import write_npz, read_npz
from . import baselines
from .mapping import AntennaMapping

__all__ = [
    "load_format",
    "VisibilityFormat",
    "has_header",
    "parse_corr_header",
    "get_header_offset",
    "format_from_header",
    "VisibilityReader",
    "discover_files",
    "write_npz",
    "read_npz",
    "baselines",
    "AntennaMapping",
]
