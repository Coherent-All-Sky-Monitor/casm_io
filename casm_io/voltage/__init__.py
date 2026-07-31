"""
casm_io.voltage — Voltage dump DADA file I/O.

Reads 4+4 bit complex voltage data from DADA files.
Supports single-subband and 3-subband (chan0_1023, chan1024_2047, chan2048_3071)
formats, and triggered dumps split over per-stream directories (stream_0 ...
stream_5).
"""

from .reader import VoltageReader
from .header import parse_dada_header
from .unpack import unpack_4bit
from .correlate import correlate

__all__ = [
    "VoltageReader",
    "parse_dada_header",
    "unpack_4bit",
    "correlate",
]
