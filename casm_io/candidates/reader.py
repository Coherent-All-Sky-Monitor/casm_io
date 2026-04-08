"""Reader for FRB search candidate files."""

import pandas as pd


# Map from Hella T1 header names to readable names
_T1_COLUMN_MAP = {
    'SNR': 'snr',
    'SAMP_START': 'sample_index',
    'TIME_START': 'time_start',
    'WIDTH': 'boxcar_width',
    'DM_IDX': 'dm_index',
    'DM': 'dm',
    'BEAM_IDX': 'beam_index',
}

_T1_INT_COLUMNS = [
    'sample_index',
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
        Path to whitespace-separated T1 output file with header row.
    """

    def __init__(self, filepath):
        self._filepath = str(filepath)
        self._df = pd.read_csv(
            filepath,
            sep=r'\s+',
        )
        self._df.rename(columns=_T1_COLUMN_MAP, inplace=True)
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
