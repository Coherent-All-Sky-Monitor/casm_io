"""Tests for SIGPROC filterbank header parsing and writing."""

import numpy as np
import pytest

from casm_io.filterbank.header import (
    read_sigproc_header,
    write_sigproc_header,
    get_frequency_axis,
    get_time_axis,
)


class TestHeaderRoundTrip:
    def test_int_float_string_roundtrip(self, tmp_path, synthetic_filterbank_header):
        """Write a header, read it back — all values must match."""
        header = synthetic_filterbank_header
        fpath = str(tmp_path / "test.fil")

        with open(fpath, "wb") as f:
            write_sigproc_header(f, header)
            # Write some dummy data so file is valid
            np.zeros(100 * 64, dtype=np.uint8).tofile(f)

        loaded, header_size = read_sigproc_header(fpath)

        assert loaded["telescope_id"] == header["telescope_id"]
        assert loaded["machine_id"] == header["machine_id"]
        assert loaded["nchans"] == header["nchans"]
        assert loaded["nbits"] == header["nbits"]
        assert loaded["source_name"] == header["source_name"]
        assert abs(loaded["tsamp"] - header["tsamp"]) < 1e-10
        assert abs(loaded["tstart"] - header["tstart"]) < 1e-6
        assert abs(loaded["fch1"] - header["fch1"]) < 1e-6
        assert abs(loaded["foff"] - header["foff"]) < 1e-10
        assert header_size > 0


class TestFrequencyAxis:
    def test_correct_values(self):
        header = {"nchans": 64, "fch1": 468.75, "foff": -0.030517578125}
        freq = get_frequency_axis(header)
        assert freq.shape == (64,)
        assert abs(freq[0] - 468.75) < 1e-6
        assert freq[1] < freq[0]  # foff is negative

    def test_ascending_foff(self):
        header = {"nchans": 10, "fch1": 100.0, "foff": 1.0}
        freq = get_frequency_axis(header)
        assert freq[0] == 100.0
        assert freq[-1] == 109.0


class TestTimeAxis:
    def test_correct_values(self):
        header = {"tsamp": 0.001}
        time = get_time_axis(header, nsamples=100)
        assert time.shape == (100,)
        assert abs(time[0]) < 1e-10
        assert abs(time[-1] - 0.099) < 1e-6
