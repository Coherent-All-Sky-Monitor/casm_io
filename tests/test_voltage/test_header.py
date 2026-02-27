"""Tests for DADA header parsing and trust mode."""

import pytest

from casm_io.voltage.header import (
    parse_dada_header,
    get_header_value,
    TRUSTED_FIELDS,
    UNTRUSTED_FIELDS,
    HEADER_SIZE,
)


class TestParseHeader:
    def test_parse_synthetic(self, synthetic_dada_file):
        fpath, expected_header, _, _ = synthetic_dada_file
        header = parse_dada_header(fpath)
        assert header["UTC_START"] == "2026-02-17-21:10:43"
        assert header["TSAMP"] == "32.768"
        assert header["NCHAN"] == "1024"
        assert header["NBIT"] == "4"
        assert header["NANT"] == "66"

    def test_header_size(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        with open(fpath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
        assert len(header_bytes) == 4096


class TestTrustMode:
    def test_untrusted_nant_returns_default(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        header = parse_dada_header(fpath)
        # Header says NANT=66, but untrusted default is "11"
        val = get_header_value(header, "NANT", trust_header=False)
        assert val == "11"

    def test_trusted_nant_returns_header_value(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        header = parse_dada_header(fpath)
        val = get_header_value(header, "NANT", trust_header=True)
        assert val == "66"

    def test_trusted_field_always_returns_header(self, synthetic_dada_file):
        """UTC_START is trusted, so trust_header=False still returns header value."""
        fpath = synthetic_dada_file[0]
        header = parse_dada_header(fpath)
        val = get_header_value(header, "UTC_START", trust_header=False)
        assert val == "2026-02-17-21:10:43"

    def test_untrusted_npol_returns_default(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        header = parse_dada_header(fpath)
        # Header says NPOL=2, default is "1"
        val = get_header_value(header, "NPOL", trust_header=False)
        assert val == "1"

    def test_file_size_uses_os(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        header = parse_dada_header(fpath)
        val = get_header_value(header, "FILE_SIZE", trust_header=False, filename=fpath)
        import os
        assert val == str(os.path.getsize(fpath))


class TestFieldSets:
    def test_no_overlap(self):
        """TRUSTED and UNTRUSTED sets must have zero overlap."""
        overlap = TRUSTED_FIELDS & UNTRUSTED_FIELDS
        assert len(overlap) == 0, f"Overlap: {overlap}"
