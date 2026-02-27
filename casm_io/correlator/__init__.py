"""
casm_io.correlator — Correlator visibility file I/O.

Two formats supported:
- pre_jan27_2026: 64 inputs (32 ant x 2 pol), 34.36s integrations
- post_jan27_2026: 128 inputs (64 ant x 2 pol), 137.44s integrations
"""

from .formats import load_format, VisibilityFormat
from .reader import read_visibilities, discover_files
from .writer import write_npz, read_npz
from . import baselines
from .mapping import AntennaMapping

__all__ = [
    "load_format",
    "VisibilityFormat",
    "read_visibilities",
    "discover_files",
    "write_npz",
    "read_npz",
    "baselines",
    "AntennaMapping",
]
