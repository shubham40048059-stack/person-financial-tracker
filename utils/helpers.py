"""
Helper functions for formatting, parsing, and data manipulation.
"""
from __future__ import annotations

import math
from typing import Any
import pandas as pd


def format_inr(value: Any, show_decimals: bool = True, default: str = "₹0.00") -> str:
    """
    Format a numeric value using the Indian Numbering System with the Rupee symbol.
    Examples:
        35246.72 -> ₹35,246.72
        103500 -> ₹1,03,500.00
        -5420.50 -> -₹5,420.50
    """
    if value is None:
        return default
    try:
        val = float(value)
    except (ValueError, TypeError):
        return default

    if math.isnan(val) or math.isinf(val):
        return default

    is_negative = val < 0
    abs_val = abs(val)

    if show_decimals:
        formatted_dec = f"{abs_val:.2f}"
        int_part, dec_part = formatted_dec.split(".")
    else:
        int_part = str(int(round(abs_val)))
        dec_part = ""

    # Indian numbering grouping: last 3 digits, then groups of 2 digits
    if len(int_part) <= 3:
        grouped_int = int_part
    else:
        last_three = int_part[-3:]
        remaining = int_part[:-3]
        # Group remaining digits in pairs of 2 from right to left
        pairs = []
        while len(remaining) > 2:
            pairs.append(remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            pairs.append(remaining)
        pairs.reverse()
        grouped_int = ",".join(pairs) + "," + last_three

    result = f"₹{grouped_int}"
    if show_decimals:
        result += f".{dec_part}"

    return f"-{result}" if is_negative else result


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely parse any value into a float, cleaning common currency characters."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value) if not math.isnan(value) else default
    try:
        cleaned = str(value).replace("₹", "").replace("$", "").replace(",", "").strip()
        if not cleaned:
            return default
        # Handle accounting negative format: (100.50)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = f"-{cleaned[1:-1]}"
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def normalize_text(text: Any) -> str:
    """Normalize text by trimming and standardizing spaces."""
    if text is None or pd.isna(text):
        return ""
    return str(text).strip()
