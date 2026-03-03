"""Reader for FRB search candidate files."""

import pandas as pd


# Hella T1 output columns -> readable names
_T1_COLUMNS = [
    'snr',
    'sample_index',
    'integration_index',
    'mjd',
    'boxcar_width',
    'dm_index',
    'dm',
    'beam_index',
]

_T1_INT_COLUMNS = [
    'sample_index',
    'integration_index',
    'boxcar_width',
    'dm_index',
    'beam_index',
]


class CandidateReader:
    """
    Reader for Hella T1 candidate lists.

    Reads the file on init (small files). Provides properties for
    quick inspection.

    Parameters
    ----------
    filepath : str or Path
        Path to whitespace-separated T1 output file (no header).
    """

    def __init__(self, filepath):
        self._filepath = str(filepath)
        self._df = pd.read_csv(
            filepath,
            sep=r'\s+',
            header=None,
            names=_T1_COLUMNS,
        )
        for col in _T1_INT_COLUMNS:
            self._df[col] = self._df[col].astype(int)

    @property
    def df(self) -> pd.DataFrame:
        """The candidate DataFrame."""
        return self._df

    @property
    def n_candidates(self) -> int:
        """Number of candidates."""
        return len(self._df)

    @property
    def snr_range(self) -> tuple[float, float]:
        """(min_snr, max_snr) range."""
        return (float(self._df['snr'].min()), float(self._df['snr'].max()))

    @property
    def dm_range(self) -> tuple[float, float]:
        """(min_dm, max_dm) range."""
        return (float(self._df['dm'].min()), float(self._df['dm'].max()))
