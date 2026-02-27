"""Tests for filterbank plotting — verify plots produce figures without crashing."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from casm_io.filterbank.plotting import (
    plot_bandpass,
    plot_timeseries,
    plot_dynamic_spectrum,
    plot_dedispersed_waterfall,
)


@pytest.fixture
def plot_data():
    """Synthetic data and header for plotting tests."""
    rng = np.random.RandomState(789)
    data = rng.randint(50, 200, size=(500, 64), dtype=np.uint8)
    header = {
        "nchans": 64,
        "fch1": 468.75,
        "foff": -0.030517578125,
        "tsamp": 0.001,
        "tstart": 60000.0,
    }
    return data, header


class TestPlotBandpass:
    def test_returns_figure(self, plot_data):
        data, header = plot_data
        fig = plot_bandpass(data, header)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path, plot_data):
        data, header = plot_data
        path = str(tmp_path / "bp.png")
        plot_bandpass(data, header, output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPlotTimeseries:
    def test_returns_figure(self, plot_data):
        data, header = plot_data
        fig = plot_timeseries(data, header)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path, plot_data):
        data, header = plot_data
        path = str(tmp_path / "ts.png")
        plot_timeseries(data, header, output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPlotDynamicSpectrum:
    def test_returns_figure(self, plot_data):
        data, header = plot_data
        fig = plot_dynamic_spectrum(data, header)
        assert isinstance(fig, plt.Figure)

    def test_with_dm(self, plot_data):
        data, header = plot_data
        # Use small DM to avoid needing too many samples
        fig = plot_dynamic_spectrum(data, header, dm=1.0)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path, plot_data):
        data, header = plot_data
        path = str(tmp_path / "ds.png")
        plot_dynamic_spectrum(data, header, output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


class TestPlotDedispersedWaterfall:
    def test_returns_figure(self, plot_data):
        data, header = plot_data
        fig = plot_dedispersed_waterfall(data, header, dm=1.0)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path, plot_data):
        data, header = plot_data
        path = str(tmp_path / "wf.png")
        plot_dedispersed_waterfall(data, header, dm=1.0, output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
