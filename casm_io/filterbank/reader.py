"""
Unified filterbank reader.

Uses sigpyproc as primary backend (fast C-backed reading).
Falls back to standalone struct-based reader if sigpyproc unavailable.

Every result includes 'backend_used' for traceability.
"""

import warnings
from pathlib import Path

import numpy as np

from .header import read_sigproc_header, get_frequency_axis, get_time_axis


class FilterbankFile:
    """
    Lazy-loading filterbank file reader.

    Header is parsed on init. Data is loaded lazily on first access
    to the `data` property.

    Parameters
    ----------
    filepath : str
        Path to .fil file.
    beam : int | None
        Beam index to load from a multibeam file. For single-beam files
        this is ignored. For multibeam files, defaults to 0 if not specified.
    verbose : bool
        Print info messages during loading.

    Attributes
    ----------
    header : dict
        Filterbank header parameters.
    backend_used : str
        'sigpyproc' or 'standalone'.
    nbeams : int
        Number of beams in the file.
    """

    def __init__(self, filepath: str, beam: int | None = None, verbose: bool = True):
        self._filepath = str(filepath)
        self._data = None
        self._verbose = verbose

        # Try sigpyproc for header, fall back to standalone
        try:
            from sigpyproc.readers import FilReader
            self._fil = FilReader(self._filepath)
            self._header = self._fil.header.to_dict()
            self._backend = "sigpyproc"

            # Compute header size and true nsamples from file size.
            # sigpyproc caps nsamples at min(header, file-derived) which
            # is wrong for multibeam files — compute from scratch.
            _, header_size = read_sigproc_header(self._filepath)
            nbeams = max(self._header.get("nbeams", 1), 1)
            nchans = self._header.get("nchans", 1)
            nbits = self._header.get("nbits", 8)
            file_size = Path(self._filepath).stat().st_size
            data_bytes = file_size - header_size
            nsamples_total = data_bytes * 8 // (nchans * nbits)
            self._header["_header_size"] = header_size
            self._header["_nsamples"] = nsamples_total // nbeams

        except ImportError:
            warnings.warn(
                "sigpyproc not available, using standalone filterbank reader",
                stacklevel=2,
            )
            self._fil = None
            header, header_size = read_sigproc_header(self._filepath)
            nchans = header.get("nchans", 1)
            nbits = header.get("nbits", 8)
            nifs = header.get("nifs", 1)
            nbeams = max(header.get("nbeams", 1), 1)
            bytes_per_sample = nchans * nifs * nbits // 8
            file_size = Path(self._filepath).stat().st_size
            data_size = file_size - header_size
            nsamples_total = data_size // bytes_per_sample
            nsamples_per_beam = nsamples_total // nbeams
            header["_nsamples"] = nsamples_per_beam
            header["_header_size"] = header_size
            self._header = header
            self._backend = "standalone"

        # Resolve beam index
        nbeams = self.nbeams
        if nbeams > 1:
            if beam is None:
                beam = 0
                if self._verbose:
                    print(f"Multibeam file ({nbeams} beams), defaulting to beam 0")
            if beam < 0 or beam >= nbeams:
                raise ValueError(
                    f"beam={beam} out of range for file with {nbeams} beams "
                    f"(valid: 0-{nbeams - 1})"
                )
        self._beam = beam

        if self._verbose:
            beam_str = f", beam {self._beam}" if nbeams > 1 else ""
            print(f"Opened {filepath} ({self._backend}): "
                  f"{self.nchans} chans, {self.nsamples} samples"
                  f"{beam_str}")

    @property
    def header(self) -> dict:
        return self._header

    @property
    def backend_used(self) -> str:
        return self._backend

    @property
    def nchans(self) -> int:
        return self._header.get("nchans", 1)

    @property
    def nbeams(self) -> int:
        return max(self._header.get("nbeams", 1), 1)

    @property
    def nsamples(self) -> int:
        return self._header["_nsamples"]

    @property
    def freq_mhz(self) -> np.ndarray:
        nchans = self.nchans
        fch1 = self._header.get("fch1", 0.0)
        foff = self._header.get("foff", 1.0)
        return fch1 + np.arange(nchans) * foff

    @property
    def time_s(self) -> np.ndarray:
        tsamp = self._header.get("tsamp", 1.0)
        return np.arange(self.nsamples) * tsamp

    @property
    def data(self) -> np.ndarray:
        """Load data on first access (lazy loading)."""
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _load_data(self) -> np.ndarray:
        if self._verbose:
            print("Loading filterbank data...")
        if self.nbeams > 1:
            data = self._load_multibeam()
        elif self._backend == "sigpyproc":
            data = self._load_sigpyproc()
        else:
            data = self._load_standalone()
        if self._verbose:
            print(f"  Loaded: {data.shape} {data.dtype}")
        return data

    def _load_sigpyproc(self) -> np.ndarray:
        block = self._fil.read_block(0, self._fil.header.nsamples)
        # sigpyproc returns (nchans, nsamples) — transpose to (nsamples, nchans)
        return block.data.T

    def _load_standalone(self) -> np.ndarray:
        header = self._header
        nchans = header.get("nchans", 1)
        nbits = header.get("nbits", 8)
        nifs = header.get("nifs", 1)
        nsamples = header["_nsamples"]
        header_size = header["_header_size"]
        expected_count = nsamples * nchans * nifs

        with open(self._filepath, "rb") as f:
            f.seek(header_size)
            if nbits == 8:
                dtype = np.int8 if header.get("signed", 0) else np.uint8
                data = np.fromfile(f, dtype=dtype, count=expected_count)
            elif nbits == 16:
                data = np.fromfile(f, dtype=np.uint16, count=expected_count)
            elif nbits == 32:
                data = np.fromfile(f, dtype=np.float32, count=expected_count)
            else:
                raise ValueError(f"Unsupported nbits: {nbits}")

        if data.size != expected_count:
            raise ValueError(
                f"Truncated filterbank data: expected {expected_count} samples, "
                f"got {data.size} in {self._filepath}"
            )

        if nifs > 1:
            data = data.reshape(nsamples, nifs, nchans)[:, 0, :]
        else:
            data = data.reshape(nsamples, nchans)
        return data

    def _load_multibeam(self) -> np.ndarray:
        """Load a single beam from a multibeam filterbank file.

        Uses raw numpy read regardless of backend to avoid sigpyproc's
        nsamples capping behavior on multibeam files.
        """
        header = self._header
        header_size = header["_header_size"]
        nbeams = self.nbeams
        nsamples = self.nsamples
        nchans = self.nchans
        nbits = header.get("nbits", 8)
        beam = self._beam

        dtype_map = {8: np.uint8, 16: np.uint16, 32: np.float32}
        if nbits == 8 and header.get("signed", 0):
            dtype = np.int8
        else:
            dtype = dtype_map.get(nbits)
            if dtype is None:
                raise ValueError(f"Unsupported nbits: {nbits}")

        total_count = nsamples * nbeams * nchans

        with open(self._filepath, "rb") as f:
            f.seek(header_size)
            raw = np.fromfile(f, dtype=dtype, count=total_count)

        if raw.size != total_count:
            raise ValueError(
                f"Truncated multibeam filterbank data: expected {total_count} "
                f"values, got {raw.size} in {self._filepath}"
            )

        data = raw.reshape(nbeams, nsamples, nchans)
        return data[beam, :, :]
