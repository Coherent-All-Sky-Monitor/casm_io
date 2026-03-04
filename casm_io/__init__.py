"""
casm_io — Unified reader and writer for all CASM data products.

Modules:
    casm_io.correlator  — Correlator visibility files (.dat)
    casm_io.voltage     — Voltage dump DADA files (.dada)
    casm_io.filterbank  — Filterbank files (.fil)
    casm_io.candidates  — FRB search candidate lists (Hella T1/T2/T3)

Quick start:
    from casm_io import VisibilityReader, load_format, AntennaMapping
    from casm_io import VoltageReader
    from casm_io import FilterbankFile, write_filterbank
    from casm_io import CandidateReader
"""

__version__ = "0.2.0"

# Convenience re-exports
from casm_io.correlator.formats import load_format, VisibilityFormat
from casm_io.correlator.mapping import AntennaMapping
from casm_io.correlator.reader import VisibilityReader
from casm_io.voltage.reader import VoltageReader
from casm_io.filterbank.reader import FilterbankFile
from casm_io.filterbank.writer import write_filterbank
from casm_io.candidates.reader import CandidateReader
from casm_io._results import VisibilityResult, SubbandResult, FullBandResult

__all__ = [
    "load_format",
    "VisibilityFormat",
    "AntennaMapping",
    "VisibilityReader",
    "VoltageReader",
    "FilterbankFile",
    "write_filterbank",
    "CandidateReader",
    "VisibilityResult",
    "SubbandResult",
    "FullBandResult",
]
