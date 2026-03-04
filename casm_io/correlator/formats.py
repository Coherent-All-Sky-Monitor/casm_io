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
