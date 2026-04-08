"""
Correlator visibility format configurations.

Each format is a JSON file describing the binary file layout:
nsig, integration time, channels, frequency setup, etc.

Built-in formats:
    "layout_32ant"  — 64 inputs (32 ant x 2 pol), 34.36s integrations
    "layout_64ant"  — 128 inputs (64 ant x 2 pol), 137.44s integrations

Custom formats: pass a JSON file path to load_format().
"""

import json
import os
from dataclasses import dataclass

import numpy as np


_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

# Map short names to JSON files
_BUILTIN_FORMATS = {
    "layout_32ant": os.path.join(_CONFIGS_DIR, "layout_32ant.json"),
    "layout_64ant": os.path.join(_CONFIGS_DIR, "layout_64ant.json"),
}


@dataclass(frozen=True)
class VisibilityFormat:
    """Configuration for a correlator visibility data format."""

    name: str
    nsig: int
    dt_raw_s: float
    ntime_per_file: int
    nchan: int
    chan_bw_mhz: float
    freq_top_mhz: float
    freq_bottom_mhz: float
    native_order: str  # "descending" or "ascending"

    @property
    def n_baselines(self) -> int:
        """Number of baselines including autos: nsig*(nsig+1)/2."""
        return self.nsig * (self.nsig + 1) // 2

    @property
    def file_duration_s(self) -> float:
        """Duration of one .dat file in seconds."""
        return self.ntime_per_file * self.dt_raw_s

    def get_frequency_axis(self, order: str = "descending") -> np.ndarray:
        """
        Build frequency axis.

        Parameters
        ----------
        order : str
            'descending' (native, highest freq first) or 'ascending' (lowest first).

        Returns
        -------
        np.ndarray
            Frequency axis in MHz with shape (nchan,).
        """
        # Native order: descending from freq_top
        freq_desc = (
            self.freq_top_mhz
            - self.chan_bw_mhz * np.arange(self.nchan, dtype=np.float64)
        )
        if order == "ascending":
            return freq_desc[::-1].copy()
        return freq_desc


    def freq_to_channel(self, freq_mhz: float) -> int:
        """
        Convert frequency in MHz to native (descending) channel index.

        Channel 0 corresponds to freq_top_mhz. Channel indices increase
        as frequency decreases (native descending order).

        Parameters
        ----------
        freq_mhz : float
            Frequency in MHz.

        Returns
        -------
        int
            Channel index in native order.

        Raises
        ------
        ValueError
            If frequency is outside the band.
        """
        if freq_mhz > self.freq_top_mhz + self.chan_bw_mhz / 2:
            raise ValueError(
                f"Frequency {freq_mhz:.4f} MHz above band top "
                f"({self.freq_top_mhz:.4f} MHz)"
            )
        if freq_mhz < self.freq_bottom_mhz - self.chan_bw_mhz / 2:
            raise ValueError(
                f"Frequency {freq_mhz:.4f} MHz below band bottom "
                f"({self.freq_bottom_mhz:.4f} MHz)"
            )
        idx = round((self.freq_top_mhz - freq_mhz) / self.chan_bw_mhz)
        return max(0, min(self.nchan - 1, idx))

    def freq_range_to_channels(
        self, freq_lo: float, freq_hi: float
    ) -> tuple[int, int]:
        """
        Convert frequency range to native (descending) channel indices.

        Parameters
        ----------
        freq_lo : float
            Lower frequency bound in MHz.
        freq_hi : float
            Upper frequency bound in MHz.

        Returns
        -------
        tuple of (int, int)
            (ch_start, ch_end) where ch_end is exclusive. In native
            descending order, ch_start corresponds to freq_hi (higher
            frequency = lower channel index).

        Raises
        ------
        ValueError
            If freq_lo >= freq_hi or range is outside the band.
        """
        if freq_lo >= freq_hi:
            raise ValueError(
                f"freq_lo ({freq_lo:.4f}) must be less than freq_hi ({freq_hi:.4f})"
            )
        # Higher freq -> lower channel index
        ch_start = self.freq_to_channel(freq_hi)
        ch_end = self.freq_to_channel(freq_lo) + 1
        ch_start = max(0, ch_start)
        ch_end = min(self.nchan, ch_end)
        return ch_start, ch_end


def load_format(name_or_path: str) -> VisibilityFormat:
    """
    Load a visibility format configuration.

    Parameters
    ----------
    name_or_path : str
        Built-in name ("layout_32ant", "layout_64ant") or path to a JSON file.

    Returns
    -------
    VisibilityFormat
    """
    if name_or_path in _BUILTIN_FORMATS:
        json_path = _BUILTIN_FORMATS[name_or_path]
    elif os.path.isfile(name_or_path):
        json_path = name_or_path
    else:
        available = list(_BUILTIN_FORMATS.keys())
        raise ValueError(
            f"Unknown format '{name_or_path}'. "
            f"Built-in formats: {available}. Or pass a JSON file path."
        )

    with open(json_path) as f:
        cfg = json.load(f)

    return VisibilityFormat(
        name=cfg["name"],
        nsig=cfg["nsig"],
        dt_raw_s=cfg["dt_raw_s"],
        ntime_per_file=cfg["ntime_per_file"],
        nchan=cfg["nchan"],
        chan_bw_mhz=cfg["chan_bw_mhz"],
        freq_top_mhz=cfg["freq_top_mhz"],
        freq_bottom_mhz=cfg["freq_bottom_mhz"],
        native_order=cfg.get("native_order", "descending"),
    )
