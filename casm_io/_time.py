"""Time conversion helpers for CASM readers."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def unix_to_datetime(unix_ts: float, tz: str = "UTC") -> datetime:
    """Convert Unix timestamp to timezone-aware datetime.

    Parameters
    ----------
    unix_ts : float
        Unix timestamp (seconds since epoch).
    tz : str
        Timezone name (e.g. 'UTC', 'America/Los_Angeles').

    Returns
    -------
    datetime
        Timezone-aware datetime.
    """
    tz_obj = timezone.utc if tz == "UTC" else ZoneInfo(tz)
    return datetime.fromtimestamp(unix_ts, tz=tz_obj)


def unix_to_iso(unix_ts: float, tz: str = "UTC") -> str:
    """Convert Unix timestamp to ISO format string in given timezone."""
    return unix_to_datetime(unix_ts, tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def format_time_span(start_unix: float, end_unix: float, tz: str = "UTC") -> str:
    """Format a time span as 'start -> end' in the given timezone."""
    return f"{unix_to_iso(start_unix, tz)} -> {unix_to_iso(end_unix, tz)}"
