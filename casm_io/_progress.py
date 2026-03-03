"""Simple inline progress bar for file/data reading loops."""

import sys


def print_progress(current, total, prefix="Reading", suffix="", width=30):
    """Print an inline progress bar using carriage return."""
    frac = current / total if total > 0 else 1.0
    filled = int(width * frac)
    bar = "=" * filled + ">" * (filled < width) + " " * (width - filled - 1)
    line = f"\r  {prefix} [{bar}] {current}/{total} {suffix}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
