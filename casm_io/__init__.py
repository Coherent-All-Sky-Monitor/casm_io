"""
casm_io — Unified reader and writer for all CASM data products.

Modules:
    casm_io.correlator  — Correlator visibility files (.dat)
    casm_io.voltage     — Voltage dump DADA files (.dada)
    casm_io.filterbank  — Filterbank files (.fil)
    casm_io.candidates  — FRB search candidate lists (Hella T1/T2/T3)

Quick start:
    from casm_io.correlator import read_visibilities, load_format, AntennaMapping
    from casm_io.voltage import read_dada_file, read_voltage_dump
    from casm_io.filterbank import read_filterbank, write_filterbank
    from casm_io.candidates import read_t1_candidates
"""

__version__ = "0.1.0"

# Convenience re-exports
from casm_io.correlator.formats import load_format, VisibilityFormat
from casm_io.correlator.mapping import AntennaMapping
from casm_io.correlator.reader import read_visibilities
from casm_io.voltage.reader import read_dada_file, read_voltage_dump
from casm_io.filterbank.reader import read_filterbank
from casm_io.filterbank.writer import write_filterbank
from casm_io.candidates.reader import read_t1_candidates
