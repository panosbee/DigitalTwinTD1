"""
Data Processing Utilities για το Digital Twin SDK.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings


def clean_cgm_data(
    data: pd.DataFrame, glucose_col: str = "cgm", timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Καθαρισμός CGM data από outliers και missing values.

    Args:
        data: Raw CGM data
        glucose_col: Column name for glucose values
        timestamp_col: Column name for timestamps

    Returns:
        Cleaned DataFrame
    """
    df = data.copy()

    # Remove impossible glucose values
    if glucose_col in df.columns:
        df = df[(df[glucose_col] >= 40) & (df[glucose_col] <= 400)]

    # Sort by timestamp
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)

    # Remove duplicates
    df = df.drop_duplicates(subset=[timestamp_col], keep="last")

    return df


def interpolate_missing_values(
    data: pd.DataFrame, glucose_col: str = "cgm", method: str = "linear"
) -> pd.DataFrame:
    """
    Interpolation για missing glucose values.
    """
    df = data.copy()

    if glucose_col in df.columns:
        df[glucose_col] = df[glucose_col].interpolate(method=method)

    return df


def create_time_features(data: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Δημιουργία time-based features.
    """
    df = data.copy()

    if timestamp_col in df.columns:
        df["hour"] = pd.to_datetime(df[timestamp_col]).dt.hour
        df["day_of_week"] = pd.to_datetime(df[timestamp_col]).dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

    return df


def calculate_glucose_statistics(data: pd.DataFrame, glucose_col: str = "cgm") -> Dict:
    """
    Υπολογισμός βασικών glucose statistics.
    """
    if glucose_col not in data.columns:
        return {}

    glucose_values = data[glucose_col].dropna()

    stats = {
        "mean": glucose_values.mean(),
        "std": glucose_values.std(),
        "min": glucose_values.min(),
        "max": glucose_values.max(),
        "time_in_range": ((glucose_values >= 70) & (glucose_values <= 180)).mean() * 100,
        "time_below_range": (glucose_values < 70).mean() * 100,
        "time_above_range": (glucose_values > 180).mean() * 100,
    }

    return stats


def detect_meal_events(
    data: pd.DataFrame, glucose_col: str = "cgm", threshold: float = 20, window_minutes: int = 60
) -> pd.DataFrame:
    """
    Ανίχνευση meal events από glucose patterns.
    """
    df = data.copy()

    if glucose_col not in df.columns:
        return df

    # Calculate glucose rate of change
    df["glucose_diff"] = df[glucose_col].diff()
    df["meal_detected"] = False

    # Simple meal detection: rapid glucose rise
    df.loc[df["glucose_diff"] > threshold, "meal_detected"] = True

    return df


# Export for testing
__all__ = [
    "clean_cgm_data",
    "interpolate_missing_values",
    "create_time_features",
    "calculate_glucose_statistics",
    "detect_meal_events",
]
