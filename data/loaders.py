"""
Φόρτωση δεδομένων από διάφορες πηγές (CGM, γεύματα, ινσουλίνη, άσκηση).
Υποστήριξη για CSV, JSON, APIs και δημοφιλή datasets όπως OhioT1DM.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, Tuple, List
import json
from pathlib import Path
import warnings


def load_data(source: Union[str, Dict], data_type: str = "auto", **kwargs) -> pd.DataFrame:
    """
    Κεντρική συνάρτηση φόρτωσης δεδομένων.

    Args:
        source: Διαδρομή αρχείου, URL ή dictionary με δεδομένα
        data_type: Τύπος δεδομένων ('cgm', 'meals', 'insulin', 'activity', 'auto')
        **kwargs: Επιπλέον παράμετροι για pandas readers

    Returns:
        pd.DataFrame: Τα φορτωμένα δεδομένα με κοινό timestamp index
    """
    if isinstance(source, str):
        if source.endswith(".csv"):
            return load_csv(source, data_type=data_type, **kwargs)
        elif source.endswith(".json"):
            return load_json(source, data_type=data_type, **kwargs)
        elif "ohio" in source.lower():
            return load_ohio_t1dm(source, **kwargs)
        else:
            # Προσπάθεια αυτόματης ανίχνευσης
            return load_csv(source, data_type=data_type, **kwargs)
    elif isinstance(source, dict):
        return pd.DataFrame(source)
    else:
        raise ValueError("Μη υποστηριζόμενος τύπος source")


def load_csv(
    filepath: str, data_type: str = "auto", timestamp_column: str = "timestamp", **kwargs
) -> pd.DataFrame:
    """
    Φόρτωση δεδομένων από CSV αρχείο.

    Args:
        filepath: Διαδρομή CSV αρχείου
        data_type: Τύπος δεδομένων
        timestamp_column: Όνομα στήλης με timestamps
        **kwargs: Επιπλέον παράμετροι για pd.read_csv

    Returns:
        pd.DataFrame: Δεδομένα με datetime index
    """
    df = pd.read_csv(filepath, **kwargs)

    # Μετατροπή timestamp σε datetime index
    if timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df.set_index(timestamp_column, inplace=True)

    # Αυτόματη ανίχνευση τύπου δεδομένων
    if data_type == "auto":
        data_type = detect_data_type(df)

    # Καθαρισμός και τυποποίηση στηλών
    df = standardize_columns(df, data_type)

    return df


def load_json(filepath: str, data_type: str = "auto", **kwargs) -> pd.DataFrame:
    """
    Φόρτωση δεδομένων από JSON αρχείο.

    Args:
        filepath: Διαδρομή JSON αρχείου
        data_type: Τύπος δεδομένων
        **kwargs: Επιπλέον παράμετροι

    Returns:
        pd.DataFrame: Δεδομένα με datetime index
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        if "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError("Μη υποστηριζόμενη δομή JSON")

    # Ανίχνευση timestamp column
    timestamp_cols = [
        col
        for col in df.columns
        if any(word in col.lower() for word in ["time", "date", "timestamp"])
    ]

    if timestamp_cols:
        df[timestamp_cols[0]] = pd.to_datetime(df[timestamp_cols[0]])
        df.set_index(timestamp_cols[0], inplace=True)

    if data_type == "auto":
        data_type = detect_data_type(df)

    df = standardize_columns(df, data_type)

    return df


def load_cgm_data(filepath: str, cgm_column: str = "glucose", **kwargs) -> pd.DataFrame:
    """
    Φόρτωση CGM (Continuous Glucose Monitor) δεδομένων.

    Args:
        filepath: Διαδρομή αρχείου με CGM δεδομένα
        cgm_column: Όνομα στήλης με τιμές γλυκόζης
        **kwargs: Επιπλέον παράμετροι

    Returns:
        pd.DataFrame: CGM δεδομένα με στανταρδοποιημένες στήλες
    """
    df = load_data(filepath, data_type="cgm", **kwargs)

    # Εύρεση στήλης γλυκόζης
    glucose_cols = [
        col
        for col in df.columns
        if any(word in col.lower() for word in ["glucose", "cgm", "bg", "sugar"])
    ]

    if glucose_cols:
        df["cgm"] = df[glucose_cols[0]]
    elif cgm_column in df.columns:
        df["cgm"] = df[cgm_column]
    else:
        raise ValueError("Δεν βρέθηκε στήλη με τιμές γλυκόζης")

    # Καθαρισμός δεδομένων CGM
    df = clean_cgm_data(df)

    return df[["cgm"]]


def load_meal_data(filepath: str, carbs_column: str = "carbs", **kwargs) -> pd.DataFrame:
    """
    Φόρτωση δεδομένων γευμάτων.

    Args:
        filepath: Διαδρομή αρχείου με δεδομένα γευμάτων
        carbs_column: Όνομα στήλης με υδατάνθρακες (g)
        **kwargs: Επιπλέον παράμετροι

    Returns:
        pd.DataFrame: Δεδομένα γευμάτων
    """
    df = load_data(filepath, data_type="meals", **kwargs)

    # Εύρεση στήλης υδατανθράκων
    carb_cols = [
        col
        for col in df.columns
        if any(word in col.lower() for word in ["carb", "cho", "carbohydrate"])
    ]

    if carb_cols:
        df["carbs"] = df[carb_cols[0]]
    elif carbs_column in df.columns:
        df["carbs"] = df[carbs_column]
    else:
        warnings.warn("Δεν βρέθηκε στήλη υδατανθράκων, δημιουργία placeholder")
        df["carbs"] = 0

    # Καθαρισμός και επεξεργασία
    df["carbs"] = pd.to_numeric(df["carbs"], errors="coerce").fillna(0)
    df = df[df["carbs"] > 0]  # Κρατάμε μόνο γεύματα με carbs

    return df[["carbs"]]


def load_insulin_data(
    filepath: str, basal_column: str = "basal", bolus_column: str = "bolus", **kwargs
) -> pd.DataFrame:
    """
    Φόρτωση δεδομένων ινσουλίνης (basal + bolus).

    Args:
        filepath: Διαδρομή αρχείου με δεδομένα ινσουλίνης
        basal_column: Όνομα στήλης με basal insulin (U/h)
        bolus_column: Όνομα στήλης με bolus insulin (U)
        **kwargs: Επιπλέον παράμετροι

    Returns:
        pd.DataFrame: Δεδομένα ινσουλίνης
    """
    df = load_data(filepath, data_type="insulin", **kwargs)

    # Αναζήτηση στηλών ινσουλίνης
    basal_cols = [
        col for col in df.columns if any(word in col.lower() for word in ["basal", "base"])
    ]
    bolus_cols = [
        col for col in df.columns if any(word in col.lower() for word in ["bolus", "meal"])
    ]

    if basal_cols:
        df["basal"] = df[basal_cols[0]]
    elif basal_column in df.columns:
        df["basal"] = df[basal_column]
    else:
        df["basal"] = 0

    if bolus_cols:
        df["bolus"] = df[bolus_cols[0]]
    elif bolus_column in df.columns:
        df["bolus"] = df[bolus_column]
    else:
        df["bolus"] = 0

    # Καθαρισμός
    df["basal"] = pd.to_numeric(df["basal"], errors="coerce").fillna(0)
    df["bolus"] = pd.to_numeric(df["bolus"], errors="coerce").fillna(0)

    return df[["basal", "bolus"]]


def load_activity_data(filepath: str, activity_column: str = "activity", **kwargs) -> pd.DataFrame:
    """
    Φόρτωση δεδομένων φυσικής δραστηριότητας.

    Args:
        filepath: Διαδρομή αρχείου με δεδομένα δραστηριότητας
        activity_column: Όνομα στήλης με δραστηριότητα
        **kwargs: Επιπλέον παράμετροι

    Returns:
        pd.DataFrame: Δεδομένα δραστηριότητας
    """
    df = load_data(filepath, data_type="activity", **kwargs)

    # Εύρεση στήλης δραστηριότητας
    activity_cols = [
        col
        for col in df.columns
        if any(word in col.lower() for word in ["activity", "exercise", "steps", "hr"])
    ]

    if activity_cols:
        df["activity"] = df[activity_cols[0]]
    elif activity_column in df.columns:
        df["activity"] = df[activity_column]
    else:
        df["activity"] = 0

    df["activity"] = pd.to_numeric(df["activity"], errors="coerce").fillna(0)

    return df[["activity"]]


def load_ohio_t1dm(dataset_path: str, subject_id: Optional[int] = None) -> pd.DataFrame:
    """
    Φόρτωση δεδομένων από το OhioT1DM dataset.

    Args:
        dataset_path: Διαδρομή του OhioT1DM dataset
        subject_id: ID συγκεκριμένου ασθενή (αν None, φορτώνει όλους)

    Returns:
        pd.DataFrame: Ενοποιημένα δεδομένα OhioT1DM
    """
    # Αυτό είναι placeholder - στην πραγματικότητα θα διαβάζει το πραγματικό dataset
    warnings.warn("Φόρτωση OhioT1DM dataset - αυτή είναι demo υλοποίηση")

    # Δημιουργία δείγματος δεδομένων για demonstration
    dates = pd.date_range("2024-01-01", periods=1440, freq="5min")  # 5 μέρες, κάθε 5 λεπτά
    n_points = len(dates)

    # Προσομοίωση CGM με φυσιολογικό πρότυπο
    base_glucose = 120 + 20 * np.sin(2 * np.pi * np.arange(n_points) / 288)  # Ημερήσιος κύκλος
    noise = np.random.normal(0, 10, n_points)
    cgm = base_glucose + noise
    cgm = np.clip(cgm, 70, 250)  # Όρια CGM

    df = pd.DataFrame(
        {
            "cgm": cgm,
            "carbs": np.random.poisson(0.1, n_points) * 20,  # Σπάνια γεύματα
            "basal": np.random.normal(1.0, 0.1, n_points),  # Basal insulin
            "bolus": np.random.exponential(0.1, n_points),  # Bolus insulin
            "activity": np.random.poisson(0.05, n_points) * 10,  # Σπάνια άσκηση
        },
        index=dates,
    )

    return df


def detect_data_type(df: pd.DataFrame) -> str:
    """
    Αυτόματη ανίχνευση τύπου δεδομένων από το DataFrame.

    Args:
        df: DataFrame για ανάλυση

    Returns:
        str: Ανιχνευμένος τύπος ('cgm', 'meals', 'insulin', 'activity', 'mixed')
    """
    columns = [col.lower() for col in df.columns]

    if any(word in " ".join(columns) for word in ["glucose", "cgm", "bg"]):
        return "cgm"
    elif any(word in " ".join(columns) for word in ["carb", "meal", "cho"]):
        return "meals"
    elif any(word in " ".join(columns) for word in ["insulin", "basal", "bolus"]):
        return "insulin"
    elif any(word in " ".join(columns) for word in ["activity", "exercise", "steps"]):
        return "activity"
    else:
        return "mixed"


def standardize_columns(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Τυποποίηση ονομάτων στηλών βάσει τύπου δεδομένων.

    Args:
        df: DataFrame για τυποποίηση
        data_type: Τύπος δεδομένων

    Returns:
        pd.DataFrame: Τυποποιημένο DataFrame
    """
    df = df.copy()
    columns_lower = {col: col.lower() for col in df.columns}

    if data_type == "cgm":
        for old_col, new_col in columns_lower.items():
            if any(word in new_col for word in ["glucose", "cgm", "bg"]):
                df.rename(columns={old_col: "cgm"}, inplace=True)
                break

    elif data_type == "meals":
        for old_col, new_col in columns_lower.items():
            if any(word in new_col for word in ["carb", "cho"]):
                df.rename(columns={old_col: "carbs"}, inplace=True)
                break

    elif data_type == "insulin":
        for old_col, new_col in columns_lower.items():
            if "basal" in new_col:
                df.rename(columns={old_col: "basal"}, inplace=True)
            elif "bolus" in new_col:
                df.rename(columns={old_col: "bolus"}, inplace=True)

    elif data_type == "activity":
        for old_col, new_col in columns_lower.items():
            if any(word in new_col for word in ["activity", "exercise", "steps"]):
                df.rename(columns={old_col: "activity"}, inplace=True)
                break

    return df


def clean_cgm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Καθαρισμός CGM δεδομένων από outliers και ελλείψεις.

    Args:
        df: DataFrame με CGM δεδομένα

    Returns:
        pd.DataFrame: Καθαρισμένα δεδομένα
    """
    df = df.copy()

    if "cgm" in df.columns:
        # Αφαίρεση outliers (εκτός φυσιολογικών ορίων)
        df["cgm"] = df["cgm"].clip(20, 600)  # mg/dL

        # Αφαίρεση ανεξήγητων άλματων
        cgm_diff = df["cgm"].diff().abs()
        outlier_threshold = cgm_diff.quantile(0.99)
        df.loc[cgm_diff > outlier_threshold, "cgm"] = np.nan

        # Interpolation για μικρά κενά (μέχρι 15 λεπτά)
        df["cgm"] = df["cgm"].interpolate(method="time", limit=3)

    return df


def combine_data_sources(
    *dataframes: pd.DataFrame, method: str = "outer", fill_method: str = "forward"
) -> pd.DataFrame:
    """
    Συνδυασμός πολλαπλών πηγών δεδομένων σε ενιαίο DataFrame.

    Args:
        *dataframes: DataFrames προς συνδυασμό
        method: Μέθοδος join ('outer', 'inner', 'left', 'right')
        fill_method: Μέθοδος πλήρωσης κενών ('forward', 'backward', 'interpolate')

    Returns:
        pd.DataFrame: Ενοποιημένα δεδομένα
    """
    if not dataframes:
        return pd.DataFrame()

    # Ξεκινάμε με το πρώτο DataFrame
    combined = dataframes[0].copy()

    # Συνδυάζουμε τα υπόλοιπα
    for df in dataframes[1:]:
        combined = combined.join(df, how=method)

    # Πλήρωση κενών
    if fill_method == "forward":
        combined = combined.fillna(method="ffill")
    elif fill_method == "backward":
        combined = combined.fillna(method="bfill")
    elif fill_method == "interpolate":
        combined = combined.interpolate(method="time")

    # Τελική πλήρωση με 0 για δεδομένα γευμάτων/ινσουλίνης
    insulin_meal_cols = ["carbs", "basal", "bolus", "activity"]
    for col in insulin_meal_cols:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    return combined
