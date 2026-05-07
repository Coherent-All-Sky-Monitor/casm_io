"""
Physical and observatory constants for CASM.
"""
import os
from dataclasses import dataclass
from pathlib import Path

# OVRO Observatory (scalars; preserved for backward compatibility)
OVRO_LAT_DEG = 37.2339
OVRO_LON_DEG = -118.2821
OVRO_ELEV_M = 1222.0


@dataclass(frozen=True)
class Site:
    """Observatory site. Use ``OVRO`` for the Owens Valley deployment."""
    lat_deg: float
    lon_deg: float
    elevation_m: float

    # Aliases used by some downstream callers.
    @property
    def alt_m(self) -> float:
        return self.elevation_m


OVRO = Site(lat_deg=OVRO_LAT_DEG, lon_deg=OVRO_LON_DEG, elevation_m=OVRO_ELEV_M)

# Frequency band (legacy values for pre-March 2026 data)
# For current frequencies, use VisibilityFormat from file headers.
# The band was shifted on 2026-03-27: FREQ_TOP changed from 468.75 to 484.375 MHz.
FREQ_TOP_MHZ = 468.75
FREQ_BOTTOM_MHZ = 375.0
N_CHAN_TOTAL = 3072
CHAN_BW_MHZ = 125.0 / 4096  # 0.030517578125 MHz

# Timing
TSAMP_US = 32.768  # Voltage dump sample time in microseconds

# Speed of light
C_LIGHT_M_S = 299792458.0
C_LIGHT_M_PER_NS = C_LIGHT_M_S * 1e-9  # ~0.299792458 m/ns

# Canonical layout-CSV resolution.
# `casm-build-layout` writes dated `casm_antenna_layout_YYYY-MM-DD.csv` files
# here and updates a `current` symlink. `AntennaMapping.load(path=None)`
# resolves: path > $CASM_LAYOUT_CSV > $CASM_LAYOUT_DIR/current.
DEFAULT_LAYOUT_DIR = Path("/home/casm/software/dev/antenna_layouts")


def resolve_layout_path(path=None) -> Path:
    """Resolve the canonical antenna-layout CSV path.

    Order:
        1. ``path`` argument if given.
        2. ``$CASM_LAYOUT_CSV`` env var if set.
        3. ``$CASM_LAYOUT_DIR/current`` symlink (default
           ``$CASM_LAYOUT_DIR = /home/casm/software/dev/antenna_layouts``).

    Raises ``FileNotFoundError`` with an actionable message if none resolve.
    """
    if path is not None:
        return Path(path)

    env_csv = os.environ.get("CASM_LAYOUT_CSV")
    if env_csv:
        return Path(env_csv)

    layout_dir = Path(os.environ.get("CASM_LAYOUT_DIR", DEFAULT_LAYOUT_DIR))
    current = layout_dir / "current"
    if current.exists():
        return current

    raise FileNotFoundError(
        f"No antenna-layout CSV found. Tried (in order):\n"
        f"  1. explicit path argument (none given)\n"
        f"  2. $CASM_LAYOUT_CSV (unset)\n"
        f"  3. {current} (no such file)\n"
        f"\n"
        f"Run `casm-build-layout` to populate {layout_dir}/current, "
        f"or set CASM_LAYOUT_CSV to point at an existing layout CSV."
    )
