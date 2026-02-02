"""Utility functions for LocalScore."""

import time
from datetime import datetime, timezone
from typing import List


def get_time_ns() -> int:
    """Get current time in nanoseconds."""
    return time.perf_counter_ns()


def get_rfc3339_timestamp() -> str:
    """Get current time in RFC 3339 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def avg(values: List[float]) -> float:
    """Calculate average of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def stdev(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = avg(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def round_to_decimal(value: float, decimals: int) -> float:
    """Round a value to specified decimal places."""
    multiplier = 10 ** decimals
    return round(value * multiplier) / multiplier


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MiB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GiB"


def format_time(time_ms: float) -> str:
    """Format time in milliseconds as human-readable string."""
    if time_ms < 1000:
        return f"{time_ms:.2f} ms"
    else:
        return f"{time_ms / 1000:.2f} s"


def sanitize_string(s: str, max_length: int = 256) -> str:
    """Sanitize and truncate string."""
    # Remove control characters
    s = "".join(c for c in s if c.isprintable() or c in "\n\t")
    return s[:max_length]


def color_str(code: str, use_color: bool = True) -> str:
    """Return ANSI color code if colors are enabled."""
    return code if use_color else ""


def join(items: List, separator: str = ", ") -> str:
    """Join items with separator, converting to strings."""
    return separator.join(str(item) for item in items)


def print_centered(width: int, char: str, text: str) -> str:
    """Create centered text with border characters."""
    content_len = len(text)
    left_pad = (width - 2 - content_len) // 2
    right_pad = width - 2 - content_len - left_pad
    return char + " " * left_pad + text + " " * right_pad + char
