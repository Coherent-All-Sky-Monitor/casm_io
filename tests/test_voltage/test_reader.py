"""Tests for voltage DADA reader."""

import numpy as np
import pytest

from casm_io.voltage.reader import read_dada_file, _guess_subband_index
from casm_io.voltage.unpack import unpack_4bit


class TestReadDadaFile:
    def test_read_shapes(self, synthetic_dada_file):
        fpath, _, raw, cfg = synthetic_dada_file
        result = read_dada_file(fpath, n_time=cfg["n_time"], verbose=False)

        # Default active_snaps from config is [0, 2, 4]
        for snap_id in [0, 2, 4]:
            v = result["voltages"][snap_id]
            assert v.shape == (cfg["n_time"], cfg["n_chan"], cfg["n_adc"])
            assert v.dtype == np.complex64

    def test_read_values_match_unpack(self, synthetic_dada_file):
        fpath, _, raw, cfg = synthetic_dada_file
        result = read_dada_file(
            fpath, n_time=cfg["n_time"], snaps=[0], freq_order="descending",
            verbose=False,
        )
        v = result["voltages"][0]

        # Compare to manual unpack of snap 0
        expected = unpack_4bit(raw[:, 0, :, :])
        np.testing.assert_array_equal(v, expected)

    def test_ascending_reverses_channels(self, synthetic_dada_file):
        fpath, _, raw, cfg = synthetic_dada_file
        desc = read_dada_file(
            fpath, n_time=cfg["n_time"], snaps=[0], freq_order="descending",
            verbose=False,
        )
        asc = read_dada_file(
            fpath, n_time=cfg["n_time"], snaps=[0], freq_order="ascending",
            verbose=False,
        )
        # Ascending should reverse the channel axis
        np.testing.assert_array_equal(
            asc["voltages"][0], desc["voltages"][0][:, ::-1, :]
        )

    def test_freq_axis_shape(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        result = read_dada_file(fpath, n_time=2, verbose=False)
        assert result["freq_mhz"].shape == (1024,)


class TestGuessSubbandIndex:
    def test_chan0_1023(self):
        assert _guess_subband_index("/data/chan0_1023/file.dada") == 0

    def test_chan1024_2047(self):
        assert _guess_subband_index("/data/chan1024_2047/file.dada") == 1

    def test_chan2048_3071(self):
        assert _guess_subband_index("/data/chan2048_3071/file.dada") == 2

    def test_unknown_defaults_to_0(self):
        assert _guess_subband_index("/data/unknown/file.dada") == 0
