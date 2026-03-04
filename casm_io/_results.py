"""Result dataclasses for all CASM readers.

Support both attribute access (result.vis) and dict-style access
(result['vis']) for backwards compatibility.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd


class _ResultBase:
    """Mixin providing dict-style access for backwards compatibility."""

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


@dataclasses.dataclass
class VisibilityResult(_ResultBase):
    """Result from VisibilityReader.read() or read_npz()."""
    vis: np.ndarray
    freq_mhz: np.ndarray
    time_unix: np.ndarray
    metadata: dict
    ref: int | None = None
    targets: np.ndarray | list | None = None


@dataclasses.dataclass
class SubbandResult(_ResultBase):
    """Result from VoltageReader.read_subband()."""
    voltages: dict
    header: dict
    freq_mhz: np.ndarray


@dataclasses.dataclass
class FullBandResult(_ResultBase):
    """Result from VoltageReader.read_full_band()."""
    voltages: np.ndarray | dict
    header: dict
    freq_mhz: np.ndarray
    utc_start: str
    antenna_df: pd.DataFrame | None
