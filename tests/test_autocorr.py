"""Tests for the voltage/visibility autocorrelation comparison tool."""

import numpy as np
import pytest

from casm_io.autocorr import autos_from_vis, voltage_autocorr
from casm_io.correlator.baselines import triu_flat_index
from casm_io.voltage.reader import VoltageReader


class TestVoltageAutocorr:
    def test_ramp_power(self, synthetic_stream_dump):
        """The ramp encodes t + 1j*stream, so |v|^2 = t^2 + stream^2."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        spec, freq, header, n_used = voltage_autocorr(
            reader, snaps=[0], gulp_samples=3, verbose=False,
        )
        assert n_used == cfg["n_time"]
        assert spec[0].shape == (3072, cfg["n_adc"])

        t_sq = np.mean(np.arange(cfg["n_time"], dtype=np.float64) ** 2)
        for index in range(6):
            block = spec[0][index * 512:(index + 1) * 512]
            if index in cfg["streams"]:
                np.testing.assert_allclose(block, t_sq + index ** 2)
            else:
                np.testing.assert_array_equal(block, 0.0)

    def test_gulped_equals_single_read(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        gulped, _, _, _ = voltage_autocorr(
            reader, snaps=[0], gulp_samples=2, verbose=False,
        )
        whole, _, _, _ = voltage_autocorr(
            reader, snaps=[0], gulp_samples=100, verbose=False,
        )
        np.testing.assert_allclose(gulped[0], whole[0])

    def test_n_time_limits_average(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        _, _, _, n_used = voltage_autocorr(
            reader, n_time=3, snaps=[0], gulp_samples=2, verbose=False,
        )
        assert n_used == 3


class TestAutosFromVis:
    def test_extracts_diagonal(self):
        nsig = 4
        n_bl = nsig * (nsig + 1) // 2
        vis = np.zeros((2, 8, n_bl), dtype=np.complex64)
        for p in range(nsig):
            vis[:, :, triu_flat_index(nsig, p, p)] = 10 * p + 1
        autos = autos_from_vis(vis, [0, 2, 3])
        assert sorted(autos) == [0, 2, 3]
        for p in autos:
            np.testing.assert_allclose(autos[p], 10 * p + 1)
            assert autos[p].shape == (8,)

    def test_time_average(self):
        nsig = 2
        vis = np.zeros((2, 4, 3), dtype=np.complex64)
        vis[0, :, 0] = 1.0
        vis[1, :, 0] = 3.0
        autos = autos_from_vis(vis, [0])
        np.testing.assert_allclose(autos[0], 2.0)

    def test_non_triangular_raises(self):
        vis = np.zeros((1, 4, 7), dtype=np.complex64)
        with pytest.raises(ValueError, match="not triangular"):
            autos_from_vis(vis, [0])

    def test_packet_index_out_of_range_raises(self):
        vis = np.zeros((1, 4, 3), dtype=np.complex64)  # nsig = 2
        with pytest.raises(ValueError, match="packet_index"):
            autos_from_vis(vis, [2])


class TestPsrdadaToUnix:
    def test_parses_fractional_seconds(self):
        from casm_io.autocorr import psrdada_to_unix
        from casm_io._time import unix_to_iso
        t = psrdada_to_unix("2026-07-30-22:25:22.958")
        assert unix_to_iso(t).startswith("2026-07-30 22:25:22")

    def test_daylight_saving_aware(self):
        """July is PDT (UTC-7), January is PST (UTC-8)."""
        from casm_io.autocorr import psrdada_to_unix
        from casm_io._time import unix_to_iso
        july = psrdada_to_unix("2026-07-30-22:00:00")
        january = psrdada_to_unix("2026-01-30-22:00:00")
        assert unix_to_iso(july, "America/Los_Angeles") == \
            "2026-07-30 15:00:00 PDT"
        assert unix_to_iso(january, "America/Los_Angeles") == \
            "2026-01-30 14:00:00 PST"

    def test_unparseable_returns_none(self):
        from casm_io.autocorr import psrdada_to_unix
        assert psrdada_to_unix("unknown") is None
        assert psrdada_to_unix(None) is None


class TestDefaultGulp:
    def test_scales_inversely_with_data_width(self):
        from casm_io.autocorr import default_gulp_samples
        narrow = default_gulp_samples(512, 12)
        wide = default_gulp_samples(3072, 132)
        assert narrow > wide > 0

    def test_mem_available_readable_on_linux(self):
        from casm_io.autocorr import mem_available_bytes
        avail = mem_available_bytes()
        assert avail is None or avail > 0
