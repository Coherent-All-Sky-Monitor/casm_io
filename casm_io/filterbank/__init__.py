"""
casm_io.filterbank — Filterbank (.fil) file I/O.

Uses sigpyproc as primary backend when available, with standalone fallback.
Every return dict includes 'backend_used' for traceability.
"""

from .reader import FilterbankFile
from .writer import write_filterbank

__all__ = [
    "FilterbankFile",
    "write_filterbank",
]
