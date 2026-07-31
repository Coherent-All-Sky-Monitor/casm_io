"""Tests for voltage DADA reader."""

import os

import numpy as np
import pytest

from casm_io.voltage.reader import (
    VoltageReader,
    _dump_sort_key,
    _guess_subband_index,
    _load_dada_config,
)
from casm_io.voltage.unpack import unpack_4bit

CHAN_BW = 0.030517578125
FREQ_TOP_STREAM = 484.375


class TestVoltageReaderSubband:
    def test_read_shapes(self, synthetic_dada_file):
        fpath, _, raw, cfg = synthetic_dada_file
        # synthetic_dada_file is in a chan0_1023/ subdirectory
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        timestamp = "2026-02-17-21:10:43"

        reader = VoltageReader(data_dir, timestamp)
        assert 0 in reader.subbands_found

        result = reader.read_subband(0, n_time=cfg["n_time"], verbose=False)

        # Default active_snaps from config is [0, 2, 4]
        for snap_id in [0, 2, 4]:
            v = result["voltages"][snap_id]
            assert v.shape == (cfg["n_time"], cfg["n_chan"], cfg["n_adc"])
            assert v.dtype == np.complex64

    def test_read_values_match_unpack(self, synthetic_dada_file):
        fpath, _, raw, cfg = synthetic_dada_file
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        timestamp = "2026-02-17-21:10:43"

        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_subband(
            0, n_time=cfg["n_time"], snaps=[0], freq_order="descending",
            verbose=False,
        )
        v = result["voltages"][0]

        # Compare to manual unpack of snap 0
        expected = unpack_4bit(raw[:, 0, :, :])
        np.testing.assert_array_equal(v, expected)

    def test_ascending_reverses_channels(self, synthetic_dada_file):
        fpath, _, raw, cfg = synthetic_dada_file
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        timestamp = "2026-02-17-21:10:43"

        reader = VoltageReader(data_dir, timestamp)
        desc = reader.read_subband(
            0, n_time=cfg["n_time"], snaps=[0], freq_order="descending",
            verbose=False,
        )
        asc = reader.read_subband(
            0, n_time=cfg["n_time"], snaps=[0], freq_order="ascending",
            verbose=False,
        )
        # Ascending should reverse the channel axis
        np.testing.assert_array_equal(
            asc["voltages"][0], desc["voltages"][0][:, ::-1, :]
        )

    def test_freq_axis_shape(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        timestamp = "2026-02-17-21:10:43"

        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_subband(0, n_time=2, verbose=False)
        assert result["freq_mhz"].shape == (1024,)

    def test_missing_subband_raises(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        timestamp = "2026-02-17-21:10:43"

        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(FileNotFoundError, match="Subband 1 not found"):
            reader.read_subband(1, verbose=False)


class TestVoltageReaderProperties:
    def test_subbands_found(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        timestamp = "2026-02-17-21:10:43"

        reader = VoltageReader(data_dir, timestamp)
        assert reader.subbands_found == [0]


class TestGuessSubbandIndex:
    def test_chan0_1023(self):
        assert _guess_subband_index("/data/chan0_1023/file.dada") == 0

    def test_chan1024_2047(self):
        assert _guess_subband_index("/data/chan1024_2047/file.dada") == 1

    def test_chan2048_3071(self):
        assert _guess_subband_index("/data/chan2048_3071/file.dada") == 2

    def test_stream_dirs(self):
        assert _guess_subband_index("/data/stream_0/file.dada") == 0
        assert _guess_subband_index("/data/stream_4/file.dada") == 4

    def test_unknown_defaults_to_0(self):
        assert _guess_subband_index("/data/unknown/file.dada") == 0


class TestDumpSortKey:
    def test_sorts_by_obs_offset(self):
        """Files order by OBS_OFFSET (bytes since UTC_START), not file number."""
        names = [
            "/d/stream_0/2026-03-16-00:30:11_0000000000270336.000000.dada",
            "/d/stream_0/2026-03-16-00:30:11_0000000000000000.000001.dada",
            "/d/stream_0/2026-03-16-00:30:11_0000000000540672.000000.dada",
        ]
        ordered = sorted(names, key=_dump_sort_key)
        offsets = [_dump_sort_key(n)[0] for n in ordered]
        assert offsets == [0, 270336, 540672]

    def test_unparsable_name_sorts_first(self):
        assert _dump_sort_key("/d/odd_name.dada")[0] == -1


# ---------------------------------------------------------------------------
# Stream layout (triggered dumps): stream_0 ... stream_5
# ---------------------------------------------------------------------------

class TestStreamConfigSelection:
    def test_stream_layout_autodetected(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        assert reader._cfg["subband_dirs"][0] == "stream_0"
        assert reader._cfg["n_subbands"] == 6
        assert reader._cfg["n_chan_per_subband"] == 512
        assert reader.subbands_found == cfg["streams"]

    def test_legacy_layout_still_autodetected(self, synthetic_dada_file):
        fpath = synthetic_dada_file[0]
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        reader = VoltageReader(data_dir, "2026-02-17-21:10:43")
        assert reader._cfg["n_subbands"] == 3
        assert reader._cfg["n_chan_per_subband"] == 1024
        assert reader._cfg["subband_dirs"][0] == "chan0_1023"

    def test_explicit_config_dict_overrides(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        legacy = _load_dada_config()
        reader = VoltageReader(data_dir, timestamp, config=legacy)
        # Legacy config looks for chan*/ dirs, so nothing is found here
        assert reader.subbands_found == []

    def test_explicit_config_path_overrides(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        import casm_io.voltage.reader as reader_mod
        path = os.path.join(
            os.path.dirname(reader_mod.__file__), "configs", "dada_format.json"
        )
        reader = VoltageReader(data_dir, timestamp, config=path)
        assert reader._cfg["n_chan_per_subband"] == 1024


class TestStreamSubbandRead:
    def test_read_shapes(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_subband(1, verbose=False)

        # Stream config activates all 11 SNAP slots
        assert sorted(result["voltages"].keys()) == list(range(cfg["n_snaps"]))
        for snap_id in range(cfg["n_snaps"]):
            v = result["voltages"][snap_id]
            assert v.shape == (cfg["n_time"], cfg["n_chan"], cfg["n_adc"])
            assert v.dtype == np.complex64

    def test_dump_bytes_limits_read(self, synthetic_stream_dump):
        """DUMP_BYTES, not the pre-allocated file size, sets the sample count."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(1, snaps=[0], verbose=False)["voltages"][0]
        assert v.shape[0] == cfg["n_time"]
        # 0xEE junk past DUMP_BYTES would unpack to -2-2j
        assert not np.any(v == np.complex64(-2 - 2j))

    def test_multi_file_time_continuity(self, synthetic_stream_dump):
        """Stream 0 is split over two files; the time ramp must survive."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(0, snaps=[0], verbose=False)["voltages"][0]

        assert v.shape[0] == cfg["n_time"]
        expected = np.arange(cfg["n_time"], dtype=np.float32).astype(np.complex64)
        np.testing.assert_array_equal(v[:, 0, 0], expected)
        # Ramp holds across every channel and ADC, including the file boundary
        for t in range(cfg["n_time"]):
            assert np.all(v[t] == np.complex64(t))

    def test_n_time_stops_inside_second_file(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        n_time = cfg["n_time_split"] + 1
        v = reader.read_subband(0, n_time=n_time, snaps=[0], verbose=False)
        v = v["voltages"][0]
        assert v.shape[0] == n_time
        np.testing.assert_array_equal(
            v[:, 0, 0], np.arange(n_time).astype(np.complex64)
        )

    def test_n_time_within_first_file(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(0, n_time=2, snaps=[0], verbose=False)
        assert v["voltages"][0].shape[0] == 2

    def test_n_time_clipped_to_available(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(0, n_time=10_000, snaps=[0], verbose=False)
        assert v["voltages"][0].shape[0] == cfg["n_time"]

    def test_subband_id_identifies_stream(self, synthetic_stream_dump):
        """Imaginary part of the ramp carries the stream index."""
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        for index in [0, 1, 3]:
            v = reader.read_subband(index, snaps=[0], verbose=False)["voltages"][0]
            assert np.all(v.imag == index)

    def test_missing_stream_raises(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(FileNotFoundError, match="Subband 2 not found"):
            reader.read_subband(2, verbose=False)

    def test_resolution_read_per_file(self, synthetic_stream_dump_43slots):
        """The March 4 epoch has 43 SNAP slots, not the config's 11."""
        data_dir, timestamp, cfg = synthetic_stream_dump_43slots
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(0, snaps=[0, 42], verbose=False)["voltages"]
        assert v[0].shape == (cfg["n_time"], cfg["n_chan"], cfg["n_adc"])
        np.testing.assert_array_equal(
            v[0][:, 0, 0], np.arange(cfg["n_time"]).astype(np.complex64)
        )

    def test_snap_beyond_file_raises(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="SNAP slots"):
            reader.read_subband(0, snaps=[cfg["n_snaps"]], verbose=False)


class TestStreamFrequencyAxis:
    def test_subband_freq_values(self, synthetic_stream_dump):
        """Stream N starts N*512 channels below the top of the band."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        for index in [0, 1, 3]:
            freq = reader.read_subband(
                index, n_time=1, snaps=[0], verbose=False,
            )["freq_mhz"]
            assert freq.shape == (512,)
            assert freq[0] == pytest.approx(FREQ_TOP_STREAM - index * 512 * CHAN_BW)
            assert freq[-1] == pytest.approx(
                FREQ_TOP_STREAM - (index * 512 + 511) * CHAN_BW
            )
            assert np.all(np.diff(freq) < 0)

    def test_stream1_starts_at_468_75(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        freq = reader.read_subband(1, n_time=1, snaps=[0], verbose=False)["freq_mhz"]
        assert freq[0] == pytest.approx(468.75)

    def test_raw_channel_mapping(self, synthetic_stream_dump):
        """Channel c of stream i is raw PFB channel 3584 - (i*512 + c)."""
        data_dir, timestamp, _ = synthetic_stream_dump
        cfg = _load_dada_config("dada_format_stream.json")
        reader = VoltageReader(data_dir, timestamp)
        for index in [0, 1, 3]:
            freq = reader.read_subband(
                index, n_time=1, snaps=[0], verbose=False,
            )["freq_mhz"]
            chans = np.arange(512)
            raw = (cfg["raw_chan_offset"] + cfg["n_chan_total"]
                   - (index * 512 + chans))
            expected = cfg["pfb_freq_bottom_mhz"] + raw * cfg["chan_bw_mhz"]
            np.testing.assert_allclose(freq, expected)

    def test_ascending_reverses_subband(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        desc = reader.read_subband(0, snaps=[0], verbose=False)
        asc = reader.read_subband(
            0, snaps=[0], freq_order="ascending", verbose=False,
        )
        np.testing.assert_array_equal(
            asc["voltages"][0], desc["voltages"][0][:, ::-1, :]
        )
        np.testing.assert_array_equal(asc["freq_mhz"], desc["freq_mhz"][::-1])

    def test_full_band_freq_endpoints(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(n_time=1, snaps=[0], verbose=False)
        freq = result["freq_mhz"]
        assert freq.shape == (3072,)
        assert freq[0] == pytest.approx(FREQ_TOP_STREAM)
        assert freq[-1] == pytest.approx(FREQ_TOP_STREAM - 3071 * CHAN_BW)

        asc = reader.read_full_band(
            n_time=1, snaps=[0], freq_order="ascending", verbose=False,
        )
        np.testing.assert_allclose(asc["freq_mhz"], freq[::-1])


class TestStreamFullBand:
    def test_full_band_shape(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(snaps=[0, 2], verbose=False)
        for snap_id in [0, 2]:
            v = result["voltages"][snap_id]
            assert v.shape == (cfg["n_time"], 3072, cfg["n_adc"])
            assert v.dtype == np.complex64
        assert result["utc_start"] == timestamp

    def test_missing_streams_zero_filled(self, synthetic_stream_dump, capsys):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_full_band(snaps=[0], verbose=False)["voltages"][0]

        for index in range(6):
            block = v[:, index * 512:(index + 1) * 512, :]
            if index in cfg["streams"]:
                assert np.all(block.imag == index)
                # Real part is the time ramp, so later samples are nonzero
                assert np.all(block[-1].real == cfg["n_time"] - 1)
            else:
                assert np.all(block == 0)

        captured = capsys.readouterr().out
        assert "zero-filling" in captured
        assert "stream_2" in captured

    def test_full_band_time_ramp_survives(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_full_band(snaps=[0], verbose=False)["voltages"][0]
        for t in range(cfg["n_time"]):
            assert np.all(v[t, :512, :].real == t)

    def test_full_band_ascending_order(self, synthetic_stream_dump):
        """Ascending puts the lowest stream first and reverses each sub-band."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        asc = reader.read_full_band(
            n_time=1, snaps=[0], freq_order="ascending", verbose=False,
        )["voltages"][0]
        # Stream 0 (top of band) now occupies the last 512 channels
        assert np.all(asc[:, -512:, :].imag == 0)
        assert np.all(asc[:, -1024:-512, :].imag == 1)
        assert np.all(asc[:, 1024:1536, :].imag == 3)
        # Streams 5, 4 (lowest) and 2 are missing
        assert np.all(asc[:, :1024, :] == 0)
        assert np.all(asc[:, 1536:2048, :] == 0)

    def test_no_stream_data_raises(self, tmp_path):
        for index in range(6):
            (tmp_path / f"stream_{index}").mkdir()
        reader = VoltageReader(str(tmp_path), "2026-03-16-00:30:11")
        assert reader.subbands_found == []
        with pytest.raises(FileNotFoundError, match="No stream data"):
            reader.read_full_band(verbose=False)


class TestStreamGapsAndAlignment:
    """Files sharing a UTC_START can belong to different dumps."""

    def test_gap_raises_by_default(self, synthetic_stream_dump_two_dumps):
        data_dir, timestamp, prefixes, cfg = synthetic_stream_dump_two_dumps
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError) as excinfo:
            reader.read_subband(0, snaps=[0], verbose=False)
        msg = str(excinfo.value)
        assert "not continuous" in msg
        # Gap reported in bytes and seconds, with both ways out
        assert "84.000 s" in msg
        assert "gap" in msg
        assert prefixes[0] in msg
        assert "allow_gaps=True" in msg

    def test_allow_gaps_stitches_and_warns(
        self, synthetic_stream_dump_two_dumps, capsys,
    ):
        data_dir, timestamp, _, cfg = synthetic_stream_dump_two_dumps
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(
            0, snaps=[0], verbose=False, allow_gaps=True,
        )["voltages"][0]

        assert v.shape[0] == 2 * cfg["n_time"]
        expected = [0, 1, 2, 4, 5, 6]
        np.testing.assert_array_equal(v[:, 0, 0].real, expected)
        assert "stitching anyway" in capsys.readouterr().out

    def test_full_prefix_selects_one_dump(self, synthetic_stream_dump_two_dumps):
        """The UTC_START_OBSOFFSET prefix picks a single dump, no gap."""
        data_dir, _, prefixes, cfg = synthetic_stream_dump_two_dumps
        for prefix, t_start in zip(prefixes, cfg["t_starts"]):
            reader = VoltageReader(data_dir, prefix)
            assert reader.subbands_found == [0]
            v = reader.read_subband(0, snaps=[0], verbose=False)["voltages"][0]
            assert v.shape[0] == cfg["n_time"]
            np.testing.assert_array_equal(
                v[:, 0, 0].real,
                np.arange(t_start, t_start + cfg["n_time"]),
            )

    def test_allow_gaps_threads_through_full_band(
        self, synthetic_stream_dump_two_dumps, capsys,
    ):
        data_dir, timestamp, _, cfg = synthetic_stream_dump_two_dumps
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="not continuous"):
            reader.read_full_band(snaps=[0], verbose=False)

        v = reader.read_full_band(
            snaps=[0], verbose=False, allow_gaps=True,
        )["voltages"][0]
        assert v.shape == (2 * cfg["n_time"], 3072, cfg["n_adc"])

    def test_stream_offset_mismatch_raises(self, synthetic_stream_dump_misaligned):
        """Streams of one dump share a first OBS_OFFSET."""
        data_dir, timestamp, offsets, _ = synthetic_stream_dump_misaligned
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError) as excinfo:
            reader.read_full_band(snaps=[0], verbose=False)
        msg = str(excinfo.value)
        assert "do not start at the same time" in msg
        for index, offset in offsets.items():
            assert f"stream_{index}" in msg
            assert str(offset) in msg

    def test_aligned_streams_do_not_raise(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(n_time=1, snaps=[0], verbose=False)
        assert result["voltages"][0].shape[1] == 3072


class TestFilledSubbands:
    def test_missing_streams_listed(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(n_time=1, snaps=[0], verbose=False)
        missing = [i for i in range(6) if i not in cfg["streams"]]
        assert result.filled_subbands == missing
        assert result["filled_subbands"] == missing

    def test_empty_when_all_streams_present(self, synthetic_stream_dump_all_streams):
        data_dir, timestamp, _ = synthetic_stream_dump_all_streams
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(snaps=[0], verbose=False)
        assert result.filled_subbands == []
        for index in range(6):
            block = result["voltages"][0][:, index * 512:(index + 1) * 512, :]
            assert np.all(block.imag == index)


class TestAntennaCsvBounds:
    def _write_csv(self, path, snap_ids, adcs):
        import pandas as pd
        pd.DataFrame({
            "antenna_id": list(range(1, len(adcs) + 1)),
            "snap_id": snap_ids,
            "adc": adcs,
            "packet_index": list(range(len(adcs))),
        }).to_csv(path, index=False)
        return str(path)

    def test_negative_adc_raises(self, synthetic_stream_dump, tmp_path):
        data_dir, timestamp, _ = synthetic_stream_dump
        csv = self._write_csv(tmp_path / "bad_adc.csv", [0, 0], [0, -1])
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError) as excinfo:
            reader.read_full_band(
                antenna_csv=csv, n_time=1, snaps=[0], verbose=False,
            )
        msg = str(excinfo.value)
        assert "row 1" in msg
        assert "adc -1" in msg

    def test_adc_above_range_raises(self, synthetic_stream_dump, tmp_path):
        data_dir, timestamp, cfg = synthetic_stream_dump
        csv = self._write_csv(
            tmp_path / "big_adc.csv", [0], [cfg["n_adc"]],
        )
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="adc"):
            reader.read_full_band(
                antenna_csv=csv, n_time=1, snaps=[0], verbose=False,
            )

    def test_negative_snap_raises(self, synthetic_stream_dump, tmp_path):
        data_dir, timestamp, _ = synthetic_stream_dump
        csv = self._write_csv(tmp_path / "bad_snap.csv", [-1], [0])
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="snap_id -1"):
            reader.read_full_band(
                antenna_csv=csv, n_time=1, snaps=[0], verbose=False,
            )

    def test_valid_csv_extracts(self, synthetic_stream_dump, tmp_path):
        data_dir, timestamp, cfg = synthetic_stream_dump
        csv = self._write_csv(tmp_path / "good.csv", [0, 0], [0, 3])
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(
            antenna_csv=csv, n_time=1, snaps=[0], verbose=False,
        )
        assert result.voltages.shape == (1, 3072, 2)
        # Stream 1 sits in channels 512-1023 and carries imag == 1
        assert np.all(result.voltages[:, 512:1024, :].imag == 1)


class TestStreamHeaderCrossCheck:
    def test_preshift_start_channel_noted(self, synthetic_stream_dump, capsys):
        """Pre-band-shift files sit one sub-band below the config grid."""
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        reader._check_stream_header(
            0, "preshift.dada", {"START_CHANNEL": "1024", "NCHAN": "512"},
        )
        out = capsys.readouterr().out
        assert "468.75" in out
        assert "512 channels apart" in out

    def test_matching_start_channel_is_quiet(self, synthetic_stream_dump, capsys):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        reader._check_stream_header(
            0, "postshift.dada",
            {"START_CHANNEL": "512", "NCHAN": "512", "STREAM_SUBBAND_ID": "0"},
        )
        assert capsys.readouterr().out == ""

    def test_wrong_subband_id_warns(self, synthetic_stream_dump, capsys):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        reader._check_stream_header(1, "x.dada", {"STREAM_SUBBAND_ID": "4"})
        assert "STREAM_SUBBAND_ID 4" in capsys.readouterr().out


class TestTimeOffset:
    def test_offset_within_first_file(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(
            0, n_time=1, snaps=[0], verbose=False, time_offset=2,
        )["voltages"][0]
        assert v.shape[0] == 1
        assert np.all(v.real == 2)

    def test_offset_spans_file_boundary(self, synthetic_stream_dump):
        """Stream 0 splits at sample 4; a read over the seam must stay continuous."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        split = cfg["n_time_split"]
        v = reader.read_subband(
            0, n_time=2, snaps=[0], verbose=False, time_offset=split - 1,
        )["voltages"][0]
        np.testing.assert_array_equal(
            v[:, 0, 0].real, [split - 1, split]
        )

    def test_offset_at_file_boundary(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        split = cfg["n_time_split"]
        v = reader.read_subband(
            0, n_time=2, snaps=[0], verbose=False, time_offset=split,
        )["voltages"][0]
        np.testing.assert_array_equal(v[:, 0, 0].real, [split, split + 1])

    def test_offset_inside_second_file(self, synthetic_stream_dump):
        """n_time=None reads whatever remains after the offset."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        offset = cfg["n_time_split"] + 1
        v = reader.read_subband(
            0, snaps=[0], verbose=False, time_offset=offset,
        )["voltages"][0]
        assert v.shape[0] == cfg["n_time"] - offset
        np.testing.assert_array_equal(
            v[:, 0, 0].real, np.arange(offset, cfg["n_time"])
        )

    def test_offset_past_end_clips_to_zero(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_subband(
            0, snaps=[0], verbose=False, time_offset=100,
        )["voltages"][0]
        assert v.shape == (0, cfg["n_chan"], cfg["n_adc"])

    def test_negative_offset_raises(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="time_offset"):
            reader.read_subband(0, snaps=[0], verbose=False, time_offset=-1)

    def test_offset_zero_matches_default(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        default = reader.read_subband(0, snaps=[0], verbose=False)
        explicit = reader.read_subband(
            0, snaps=[0], verbose=False, time_offset=0,
        )
        np.testing.assert_array_equal(
            default["voltages"][0], explicit["voltages"][0]
        )

    def test_gulps_concatenate_to_full_read(self, synthetic_stream_dump):
        """Reading in gulps must reproduce the single full-band read exactly."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        full = reader.read_full_band(snaps=[0], verbose=False)["voltages"][0]

        gulp = 3
        parts = []
        for offset in range(0, cfg["n_time"], gulp):
            res = reader.read_full_band(
                snaps=[0], n_time=gulp, time_offset=offset, verbose=False,
            )
            parts.append(res["voltages"][0])
        np.testing.assert_array_equal(np.concatenate(parts, axis=0), full)

    def test_offset_full_band(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        v = reader.read_full_band(
            snaps=[0], n_time=2, time_offset=2, verbose=False,
        )["voltages"][0]
        assert v.shape == (2, 3072, cfg["n_adc"])
        # Present streams carry the offset time ramp; missing ones are zeros
        np.testing.assert_array_equal(v[:, 0, 0].real, [2, 3])

    def test_offset_legacy_layout(self, synthetic_dada_file):
        """Legacy single-file layout honours the offset too."""
        fpath, _, raw, cfg = synthetic_dada_file
        data_dir = str(fpath).rsplit("/chan0_1023/", 1)[0]
        reader = VoltageReader(data_dir, "2026-02-17-21:10:43")
        v = reader.read_subband(
            0, snaps=[0], verbose=False, time_offset=2,
        )["voltages"][0]
        assert v.shape[0] == cfg["n_time"] - 2
        np.testing.assert_array_equal(v, unpack_4bit(raw[2:, 0, :, :]))


class TestSubbandSelection:
    def test_subset_shapes_and_freq(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(
            snaps=[0], subbands=[0, 1], verbose=False,
        )
        v = result["voltages"][0]
        assert v.shape == (cfg["n_time"], 1024, cfg["n_adc"])
        assert np.all(v[:, :512, :].imag == 0)
        assert np.all(v[:, 512:, :].imag == 1)
        freq = result["freq_mhz"]
        assert freq.shape == (1024,)
        assert freq[0] == pytest.approx(FREQ_TOP_STREAM)
        assert freq[-1] == pytest.approx(FREQ_TOP_STREAM - 1023 * CHAN_BW)

    def test_single_subband_freq_offset(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(
            snaps=[0], subbands=[3], verbose=False,
        )
        assert result["voltages"][0].shape[1] == 512
        assert result["freq_mhz"][0] == pytest.approx(
            FREQ_TOP_STREAM - 3 * 512 * CHAN_BW
        )
        assert np.all(result["voltages"][0].imag == 3)

    def test_filled_only_within_selection(self, synthetic_stream_dump):
        """Stream 2 is missing; streams outside the selection are not filled."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(
            snaps=[0], subbands=[1, 2], verbose=False,
        )
        assert result.filled_subbands == [2]
        v = result["voltages"][0]
        assert np.all(v[:, :512, :].imag == 1)
        assert np.all(v[:, 512:, :] == 0)

    def test_gap_selection_raises(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="contiguous"):
            reader.read_full_band(snaps=[0], subbands=[0, 2], verbose=False)

    def test_unsorted_selection_raises(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="contiguous"):
            reader.read_full_band(snaps=[0], subbands=[1, 0], verbose=False)

    def test_empty_selection_raises(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="contiguous"):
            reader.read_full_band(snaps=[0], subbands=[], verbose=False)

    def test_out_of_range_selection_raises(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="within 0-5"):
            reader.read_full_band(snaps=[0], subbands=[5, 6], verbose=False)

    def test_selection_with_no_data_raises(self, synthetic_stream_dump):
        """Streams 4 and 5 have no files; selecting only them must raise."""
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(FileNotFoundError, match="No stream data"):
            reader.read_full_band(snaps=[0], subbands=[4, 5], verbose=False)

    def test_subset_with_csv(self, synthetic_stream_dump, tmp_path):
        import pandas as pd
        data_dir, timestamp, cfg = synthetic_stream_dump
        csv = tmp_path / "two_ants.csv"
        pd.DataFrame({
            "antenna_id": [1, 2],
            "snap_id": [0, 0],
            "adc": [0, 3],
            "packet_index": [0, 3],
        }).to_csv(csv, index=False)
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(
            antenna_csv=str(csv), snaps=[0], subbands=[1], verbose=False,
        )
        assert result.voltages.shape == (cfg["n_time"], 512, 2)
        assert np.all(result.voltages.imag == 1)

    def test_ascending_subset(self, synthetic_stream_dump):
        """Ascending order flips the selected sub-bands too."""
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        result = reader.read_full_band(
            snaps=[0], subbands=[0, 1], freq_order="ascending", verbose=False,
        )
        v = result["voltages"][0]
        assert np.all(v[:, :512, :].imag == 1)
        assert np.all(v[:, 512:, :].imag == 0)
        freq = result["freq_mhz"]
        assert np.all(np.diff(freq) > 0)
        assert freq[-1] == pytest.approx(FREQ_TOP_STREAM)

    def test_misaligned_stream_outside_selection_skipped(
        self, synthetic_stream_dump_misaligned,
    ):
        """A bad stream outside the selection cannot block the read."""
        data_dir, timestamp, offsets, cfg = synthetic_stream_dump_misaligned
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="do not start at the same time"):
            reader.read_full_band(snaps=[0], subbands=[0, 1], verbose=False)

        result = reader.read_full_band(snaps=[0], subbands=[0], verbose=False)
        v = result["voltages"][0]
        assert v.shape == (cfg["n_time"], 512, cfg["n_adc"])
        assert np.all(v.imag == 0)


class TestSecondsArguments:
    def test_seconds_maps_to_samples(self, synthetic_stream_dump):
        from casm_io.constants import TSAMP_S
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        by_samples = reader.read_full_band(n_time=3, snaps=[0], verbose=False)
        by_seconds = reader.read_full_band(
            seconds=3 * TSAMP_S, snaps=[0], verbose=False,
        )
        np.testing.assert_array_equal(
            by_samples["voltages"][0], by_seconds["voltages"][0]
        )

    def test_offset_seconds_maps_to_samples(self, synthetic_stream_dump):
        from casm_io.constants import TSAMP_S
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        a = reader.read_full_band(time_offset=2, snaps=[0], verbose=False)
        b = reader.read_full_band(
            offset_seconds=2 * TSAMP_S, snaps=[0], verbose=False,
        )
        np.testing.assert_array_equal(a["voltages"][0], b["voltages"][0])

    def test_both_forms_raise(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        with pytest.raises(ValueError, match="not both"):
            reader.read_full_band(n_time=1, seconds=1.0, snaps=[0], verbose=False)
        with pytest.raises(ValueError, match="not both"):
            reader.read_full_band(time_offset=1, offset_seconds=1.0,
                                  snaps=[0], verbose=False)


class TestIterFullBand:
    def test_chunks_concatenate_to_full_read(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        full = reader.read_full_band(snaps=[0], verbose=False)["voltages"][0]
        chunks = [c["voltages"][0] for c in
                  reader.iter_full_band(gulp_samples=3, snaps=[0])]
        assert len(chunks) == 3  # 7 samples in gulps of 3
        np.testing.assert_array_equal(np.concatenate(chunks, axis=0), full)

    def test_n_time_bounds_the_iteration(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        total = sum(c["voltages"][0].shape[0] for c in
                    reader.iter_full_band(n_time=5, gulp_samples=2, snaps=[0]))
        assert total == 5

    def test_offset_and_kwargs_pass_through(self, synthetic_stream_dump):
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        chunks = list(reader.iter_full_band(
            time_offset=5, gulp_samples=10, snaps=[0], subbands=[0, 1],
        ))
        assert len(chunks) == 1
        v = chunks[0]["voltages"][0]
        assert v.shape == (cfg["n_time"] - 5, 1024, cfg["n_adc"])
        np.testing.assert_array_equal(v[:, 0, 0].real, [5, 6])

    def test_stops_cleanly_past_the_end(self, synthetic_stream_dump):
        data_dir, timestamp, _ = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        assert list(reader.iter_full_band(time_offset=100, gulp_samples=2,
                                          snaps=[0])) == []

    def test_default_gulp_comes_from_ram(self, synthetic_stream_dump):
        """No gulp given: the whole 7-sample dump fits in one RAM-sized chunk."""
        data_dir, timestamp, cfg = synthetic_stream_dump
        reader = VoltageReader(data_dir, timestamp)
        chunks = list(reader.iter_full_band(snaps=[0]))
        assert len(chunks) == 1
        assert chunks[0]["voltages"][0].shape[0] == cfg["n_time"]
