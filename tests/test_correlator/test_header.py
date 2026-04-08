"""Tests for correlator visibility file header parsing."""

import os

import numpy as np
import pytest

from casm_io.correlator.header import (
    has_header,
    parse_corr_header,
    get_header_offset,
    format_from_header,
    HEADER_SIZE,
)
from casm_io.correlator.reader import VisibilityReader


class TestHasHeader:
    def test_detects_header(self, synthetic_dat_with_header):
        fpath, _, _, _ = synthetic_dat_with_header
        assert has_header(fpath) is True

    def test_no_header(self, tiny_dat_file):
        fpath, _ = tiny_dat_file
        assert has_header(fpath) is False


class TestParseCorrelatorHeader:
    def test_parses_all_fields(self, synthetic_dat_with_header):
        fpath, expected_header, _, _ = synthetic_dat_with_header
        header = parse_corr_header(fpath)
        assert header["HDR_SIZE"] == "4096"
        assert header["NCHAN"] == expected_header["NCHAN"]
        assert header["NBASELINE"] == expected_header["NBASELINE"]
        assert header["CORR_DUMP_DUMPS_PER_FILE"] == expected_header["CORR_DUMP_DUMPS_PER_FILE"]
        assert header["TSAMP"] == expected_header["TSAMP"]
        assert header["FREQ_START"] == expected_header["FREQ_START"]
        assert header["CHANBW"] == expected_header["CHANBW"]
        assert header["UTC_START"] == "2026-03-05-08:02:39"

    def test_ignores_null_padding(self, synthetic_dat_with_header):
        fpath, _, _, _ = synthetic_dat_with_header
        header = parse_corr_header(fpath)
        # No keys should contain null bytes
        for k, v in header.items():
            assert "\x00" not in k
            assert "\x00" not in v


class TestFormatFromHeader:
    def test_builds_format(self, synthetic_dat_with_header):
        fpath, _, _, fmt = synthetic_dat_with_header
        header = parse_corr_header(fpath)
        built = format_from_header(header)
        assert built.nchan == fmt.nchan
        assert built.n_baselines == fmt.n_baselines
        assert built.nsig == fmt.nsig
        assert built.ntime_per_file == fmt.ntime_per_file
        assert abs(built.dt_raw_s - fmt.dt_raw_s) < 1e-12
        assert abs(built.freq_top_mhz - fmt.freq_top_mhz) < 1e-6
        assert abs(built.chan_bw_mhz - fmt.chan_bw_mhz) < 1e-12
        assert built.native_order == "descending"

    def test_invalid_nbaseline(self):
        header = {
            "NCHAN": "8",
            "NBASELINE": "7",  # Not a valid triangular number
            "CORR_DUMP_DUMPS_PER_FILE": "2",
            "TSAMP": "1000000000000",
            "FREQ_START": "468.75",
            "CHANBW": "-0.030517578125",
        }
        with pytest.raises(ValueError, match="does not correspond"):
            format_from_header(header)

    def test_layout_64ant_values(self):
        """Verify header values matching layout_64ant.json produce correct format."""
        header = {
            "NCHAN": "3072",
            "NBASELINE": "8256",
            "CORR_DUMP_DUMPS_PER_FILE": "32",
            "TSAMP": "137438953.472",  # microseconds
            "FREQ_START": "468.75",
            "CHANBW": "-0.030517578125",
        }
        fmt = format_from_header(header)
        assert fmt.nsig == 128
        assert fmt.nchan == 3072
        assert fmt.n_baselines == 8256
        assert fmt.ntime_per_file == 32
        assert abs(fmt.dt_raw_s - 137.438953472) < 1e-9
        assert abs(fmt.freq_top_mhz - 468.75) < 1e-6
        assert abs(fmt.freq_bottom_mhz - 375.0) < 1e-3


class TestGetHeaderOffset:
    def test_with_header(self, synthetic_dat_with_header):
        fpath, _, _, _ = synthetic_dat_with_header
        offset, header = get_header_offset(fpath)
        assert offset == HEADER_SIZE
        assert header is not None
        assert "NCHAN" in header

    def test_without_header(self, tiny_dat_file):
        fpath, _ = tiny_dat_file
        offset, header = get_header_offset(fpath)
        assert offset == 0
        assert header is None


class TestReaderWithHeader:
    def test_reads_headered_file(self, tmp_path, synthetic_dat_with_header, tiny_format):
        fpath, _, data, fmt = synthetic_dat_with_header
        data_dir = str(tmp_path)
        reader = VisibilityReader(data_dir, "2026-03-05-08:02:39", fmt)
        result = reader.read(nfiles=1, verbose=False)
        assert result.vis.shape == (fmt.ntime_per_file, fmt.nchan, fmt.n_baselines)
        # Verify data matches (first time step, first channel, first baseline)
        assert result.vis[0, 0, 0].real == pytest.approx(0.0)
        assert result.vis[0, 0, 0].imag == pytest.approx(1.0)

    def test_backward_compat_no_header(self, tmp_path, tiny_dat_file, tiny_format):
        """Old files without header still work with explicit fmt."""
        fpath, _ = tiny_dat_file
        data_dir = str(tmp_path)
        reader = VisibilityReader(data_dir, "2026-01-01-00:00:00", tiny_format)
        result = reader.read(nfiles=1, verbose=False)
        assert result.vis.shape == (tiny_format.ntime_per_file, tiny_format.nchan, tiny_format.n_baselines)

    def test_file_headers_in_metadata(self, tmp_path, synthetic_dat_with_header, tiny_format):
        fpath, _, _, fmt = synthetic_dat_with_header
        reader = VisibilityReader(str(tmp_path), "2026-03-05-08:02:39", fmt)
        result = reader.read(nfiles=1, verbose=False)
        assert "file_headers" in result.metadata
        assert 0 in result.metadata["file_headers"]
        assert result.metadata["file_headers"][0]["NCHAN"] == str(fmt.nchan)


class TestReaderAutoFormat:
    def test_auto_format_from_header(self, tmp_path, synthetic_dat_with_header):
        """fmt=None works for headered files."""
        fpath, _, data, fmt = synthetic_dat_with_header
        reader = VisibilityReader(str(tmp_path), "2026-03-05-08:02:39")
        result = reader.read(nfiles=1, verbose=False)
        assert result.vis.shape == (fmt.ntime_per_file, fmt.nchan, fmt.n_baselines)

    def test_auto_format_errors_for_old_files(self, tmp_path, tiny_dat_file):
        """fmt=None raises for old files without header."""
        fpath, _ = tiny_dat_file
        with pytest.raises(ValueError, match="No header found"):
            VisibilityReader(str(tmp_path), "2026-01-01-00:00:00")
class TestHeaderFreqChange:
    """Tests for the March 27 2026 frequency band shift."""

    def test_new_freq_start_484(self, tmp_path):
        """Header with FREQ_START=484.375 should produce correct format."""
        header_lines = [
            "HDR_SIZE 4096",
            "NCHAN 3072",
            "NBASELINE 8256",
            "CORR_DUMP_DUMPS_PER_FILE 32",
            "TSAMP 137438953.472",
            "FREQ_START 484.375",
            "CHANBW -0.030517578125",
        ]
        header_text = "\n".join(header_lines) + "\n"
        header_bytes = header_text.encode("ascii").ljust(4096, b"\x00")
        fpath = tmp_path / "test_header.dat"
        fpath.write_bytes(header_bytes)

        from casm_io.correlator.header import parse_corr_header, format_from_header
        header = parse_corr_header(str(fpath))
        fmt = format_from_header(header)
        assert abs(fmt.freq_top_mhz - 484.375) < 0.001
        assert abs(fmt.freq_bottom_mhz - (484.375 - 3072 * 0.030517578125)) < 0.01

    def test_native_order_from_chanbw_sign(self):
        """native_order should be derived from CHANBW sign, not hardcoded."""
        from casm_io.correlator.header import format_from_header

        header_desc = {
            "NCHAN": "8", "NBASELINE": "10", "CORR_DUMP_DUMPS_PER_FILE": "2",
            "TSAMP": "1000000", "FREQ_START": "468.75", "CHANBW": "-0.030517578125",
        }
        fmt = format_from_header(header_desc)
        assert fmt.native_order == "descending"

        header_asc = dict(header_desc)
        header_asc["CHANBW"] = "0.030517578125"
        fmt_asc = format_from_header(header_asc)
        assert fmt_asc.native_order == "ascending"


class TestReaderFreqStartOverride:
    """Test that VisibilityReader overrides fmt.freq_top_mhz from file header."""

    def test_explicit_fmt_overridden_by_header_freq_start(self, tmp_path):
        """When fmt has stale freq_top_mhz, header FREQ_START should win."""
        nsig = 4
        nchan = 8
        nbl = nsig * (nsig + 1) // 2  # 10
        ntime = 2
        chan_bw = 0.030517578125
        old_freq_top = 468.75
        new_freq_top = 484.375

        # Build a file with header saying FREQ_START=484.375
        header_lines = [
            "HDR_SIZE 4096",
            f"NCHAN {nchan}",
            f"NBASELINE {nbl}",
            f"CORR_DUMP_DUMPS_PER_FILE {ntime}",
            "TSAMP 1000000",
            f"FREQ_START {new_freq_top}",
            f"CHANBW -{chan_bw}",
            "UTC_START 2026-03-28-12:00:00",
        ]
        header_text = "\n".join(header_lines) + "\n"
        header_bytes = header_text.encode("ascii").ljust(4096, b"\x00")

        data = np.zeros((ntime, nchan, nbl, 2), dtype=np.int32)
        fpath = tmp_path / "2026-03-28-12:00:00.0"
        with open(fpath, "wb") as f:
            f.write(header_bytes)
            data.tofile(f)

        # Explicit fmt with old freq_top_mhz
        from casm_io.correlator.formats import VisibilityFormat
        old_fmt = VisibilityFormat(
            name="test_old",
            nsig=nsig,
            dt_raw_s=1.0,
            ntime_per_file=ntime,
            nchan=nchan,
            chan_bw_mhz=chan_bw,
            freq_top_mhz=old_freq_top,
            freq_bottom_mhz=old_freq_top - nchan * chan_bw,
            native_order="descending",
        )

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reader = VisibilityReader(str(tmp_path), "2026-03-28-12:00:00", old_fmt)
            # Should have emitted a warning about the mismatch
            freq_warnings = [x for x in w if "FREQ_START" in str(x.message)]
            assert len(freq_warnings) == 1
            assert "484.375" in str(freq_warnings[0].message)

        # The reader's format should use the header value
        result = reader.read(nfiles=1, verbose=False)
        assert abs(result.freq_mhz[0] - new_freq_top) < 0.001

    def test_explicit_fmt_no_override_when_matching(self, tmp_path):
        """No warning when header FREQ_START matches fmt.freq_top_mhz."""
        nsig = 4
        nchan = 8
        nbl = nsig * (nsig + 1) // 2
        ntime = 2
        chan_bw = 0.030517578125
        freq_top = 468.75

        header_lines = [
            "HDR_SIZE 4096",
            f"NCHAN {nchan}",
            f"NBASELINE {nbl}",
            f"CORR_DUMP_DUMPS_PER_FILE {ntime}",
            "TSAMP 1000000",
            f"FREQ_START {freq_top}",
            f"CHANBW -{chan_bw}",
            "UTC_START 2026-03-05-08:00:00",
        ]
        header_text = "\n".join(header_lines) + "\n"
        header_bytes = header_text.encode("ascii").ljust(4096, b"\x00")

        data = np.zeros((ntime, nchan, nbl, 2), dtype=np.int32)
        fpath = tmp_path / "2026-03-05-08:00:00.0"
        with open(fpath, "wb") as f:
            f.write(header_bytes)
            data.tofile(f)

        from casm_io.correlator.formats import VisibilityFormat
        fmt = VisibilityFormat(
            name="test_match",
            nsig=nsig,
            dt_raw_s=1.0,
            ntime_per_file=ntime,
            nchan=nchan,
            chan_bw_mhz=chan_bw,
            freq_top_mhz=freq_top,
            freq_bottom_mhz=freq_top - nchan * chan_bw,
            native_order="descending",
        )

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reader = VisibilityReader(str(tmp_path), "2026-03-05-08:00:00", fmt)
            freq_warnings = [x for x in w if "FREQ_START" in str(x.message)]
            assert len(freq_warnings) == 0

    def test_explicit_fmt_no_header_falls_back(self, tmp_path, tiny_format):
        """Old files without header: explicit fmt used as-is, no warning."""
        fmt = tiny_format
        nbl = fmt.n_baselines
        data = np.zeros((fmt.ntime_per_file, fmt.nchan, nbl, 2), dtype=np.int32)
        fpath = tmp_path / "2026-01-01-00:00:00.0"
        data.tofile(str(fpath))

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reader = VisibilityReader(str(tmp_path), "2026-01-01-00:00:00", fmt)
            freq_warnings = [x for x in w if "FREQ_START" in str(x.message)]
            assert len(freq_warnings) == 0

        result = reader.read(nfiles=1, verbose=False)
        assert abs(result.freq_mhz[0] - fmt.freq_top_mhz) < 0.001

