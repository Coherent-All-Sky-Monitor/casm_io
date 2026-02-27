"""Tests for 4+4 bit complex unpacking."""

import numpy as np
import pytest

from casm_io.voltage.unpack import unpack_4bit


class TestUnpack4bit:
    """Each byte: upper 4 bits = real (signed), lower 4 bits = imag (signed)."""

    def test_zero(self):
        data = np.array([0x00], dtype=np.uint8)
        result = unpack_4bit(data)
        assert result[0] == 0 + 0j

    def test_positive_both(self):
        # 0x73 = 0111 0011 → real=7, imag=3
        data = np.array([0x73], dtype=np.uint8)
        result = unpack_4bit(data)
        assert result[0] == 7 + 3j

    def test_negative_upper_positive_lower(self):
        # 0xF1 = 1111 0001 → real=15→-1, imag=1
        data = np.array([0xF1], dtype=np.uint8)
        result = unpack_4bit(data)
        assert result[0] == -1 + 1j

    def test_min_value_negative_lower(self):
        # 0x8E = 1000 1110 → real=8→-8, imag=14→-2
        data = np.array([0x8E], dtype=np.uint8)
        result = unpack_4bit(data)
        assert result[0] == -8 - 2j

    def test_all_ones(self):
        # 0xFF = 1111 1111 → real=15→-1, imag=15→-1
        data = np.array([0xFF], dtype=np.uint8)
        result = unpack_4bit(data)
        assert result[0] == -1 - 1j

    def test_output_range(self):
        """All 256 possible bytes produce values in [-8, +7] for both real and imag."""
        data = np.arange(256, dtype=np.uint8)
        result = unpack_4bit(data)
        assert np.all(result.real >= -8)
        assert np.all(result.real <= 7)
        assert np.all(result.imag >= -8)
        assert np.all(result.imag <= 7)

    def test_output_dtype(self):
        data = np.array([0x00, 0xFF], dtype=np.uint8)
        result = unpack_4bit(data)
        assert result.dtype == np.complex64

    def test_preserves_shape(self):
        data = np.zeros((3, 4, 5), dtype=np.uint8)
        result = unpack_4bit(data)
        assert result.shape == (3, 4, 5)
