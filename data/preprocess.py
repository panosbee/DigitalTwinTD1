"""
Προεπεξεργασία και feature engineering για δεδομένα ψηφιακού διδύμου.
Περιλαμβάνει καθαρισμό, ευθυγράμμιση χρονοσειρών και δημιουργία χαρακτηριστικών.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


class T1DDataPreprocessor:
    """
    Κλάση για προεπεξεργασία δεδομένων διαβήτη τύπου 1.

    Παρέχει μεθόδους για:
    - Καθαρισμό και ευθυγράμμιση χρονοσειρών
    - Feature engineering (sliding windows, lag features)
    - Κανονικοποίηση δεδομένων
    - Χειρισμό ελλειπόντων τιμών
    """

    def __init__(
        self,
        sampling_rate: str = "5min",
        cgm_limits: Tuple[float, float] = (40, 400),
        interpolation_limit: int = 3,
    ):
        """
        Αρχικοποίηση preprocessor.

        Args:
            sampling_rate: Συχνότητα δειγματοληψίας (π.χ. '5min', '1min')
            cgm_limits: Όρια CGM τιμών (min, max) σε mg/dL
            interpolation_limit: Μέγιστος αριθμός συνεχόμενων NaN για interpolation
        """
        self.sampling_rate = sampling_rate
        self.cgm_limits = cgm_limits
        self.interpolation_limit = interpolation_limit
        self.scalers = {}
        self.feature_columns = None

    def fit_transform(self, data: pd.DataFrame, target_column: str = "cgm") -> pd.DataFrame:
        """
        Προεπεξεργασία δεδομένων (fit + transform).

        Args:
            data: Ακατέργαστα δεδομένα
            target_column: Όνομα στήλης target (γλυκόζη)

        Returns:
            pd.DataFrame: Προεπεξεργασμένα δεδομένα
        """
        # Βήμα 1: Καθαρισμός δεδομένων
        clean_data = self.clean_data(data)

        # Βήμα 2: Ευθυγράμμιση χρονοσειρών
        aligned_data = self.align_time_series(clean_data)

        # Βήμα 3: Feature engineering
        featured_data = self.create_features(aligned_data, target_column)

        # Βήμα 4: Κανονικοποίηση
        normalized_data = self.normalize_features(featured_data, fit=True)

        return normalized_data

    def transform(self, data: pd.DataFrame, target_column: str = "cgm") -> pd.DataFrame:
        """
        Εφαρμογή προεπεξεργασίας σε νέα δεδομένα (χωρίς fit).

        Args:
            data: Νέα δεδομένα
            target_column: Όνομα στήλης target

        Returns:
            pd.DataFrame: Προεπεξεργασμένα δεδομένα
        """
        clean_data = self.clean_data(data)
        aligned_data = self.align_time_series(clean_data)
        featured_data = self.create_features(aligned_data, target_column)
        normalized_data = self.normalize_features(featured_data, fit=False)

        return normalized_data

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Καθαρισμός δεδομένων από outliers και ανωμαλίες.

        Args:
            data: Ακατέργαστα δεδομένα

        Returns:
            pd.DataFrame: Καθαρισμένα δεδομένα
        """
        data = data.copy()

        # Καθαρισμός CGM δεδομένων
        if "cgm" in data.columns:
            # Όρια CGM
            data["cgm"] = data["cgm"].clip(self.cgm_limits[0], self.cgm_limits[1])

            # Αφαίρεση ανεξήγητων άλματων (>100 mg/dL σε 5 λεπτά)
            cgm_diff = data["cgm"].diff().abs()
            outlier_mask = cgm_diff > 100
            data.loc[outlier_mask, "cgm"] = np.nan

            # Αφαίρεση συνεχόμενων ίδιων τιμών (sensor error)
            consecutive_same = (data["cgm"].diff() == 0).rolling(window=6).sum() >= 5
            data.loc[consecutive_same, "cgm"] = np.nan

        # Καθαρισμός ινσουλίνης (αρνητικές τιμές)
        insulin_cols = ["basal", "bolus"]
        for col in insulin_cols:
            if col in data.columns:
                data[col] = data[col].clip(lower=0)

        # Καθαρισμός υδατανθράκων
        if "carbs" in data.columns:
            data["carbs"] = data["carbs"].clip(lower=0, upper=200)  # Μέγιστο 200g

        return data

    def align_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ευθυγράμμιση χρονοσειρών σε κοινή συχνότητα.

        Args:
            data: Δεδομένα προς ευθυγράμμιση

        Returns:
            pd.DataFrame: Ευθυγραμμισμένα δεδομένα
        """
        # Resampling σε καθορισμένη συχνότητα
        data_resampled = data.resample(self.sampling_rate).agg(
            {"cgm": "mean", "carbs": "sum", "basal": "mean", "bolus": "sum", "activity": "mean"}
        )

        # Interpolation για CGM και basal (συνεχείς μετρήσεις)
        continuous_cols = ["cgm", "basal", "activity"]
        for col in continuous_cols:
            if col in data_resampled.columns:
                data_resampled[col] = data_resampled[col].interpolate(
                    method="time", limit=self.interpolation_limit
                )

        # Πλήρωση με 0 για discrete events (γεύματα, bolus)
        event_cols = ["carbs", "bolus"]
        for col in event_cols:
            if col in data_resampled.columns:
                data_resampled[col] = data_resampled[col].fillna(0)

        return data_resampled

    def create_features(self, data: pd.DataFrame, target_column: str = "cgm") -> pd.DataFrame:
        """
        Δημιουργία χαρακτηριστικών για machine learning.

        Args:
            data: Ευθυγραμμισμένα δεδομένα
            target_column: Στήλη target

        Returns:
            pd.DataFrame: Δεδομένα με πρόσθετα features
        """
        data = data.copy()

        # Lag features για CGM (τιμές παρελθόντος)
        if target_column in data.columns:
            for lag in [1, 2, 3, 6, 12]:  # 5, 10, 15, 30, 60 λεπτά πίσω
                data[f"{target_column}_lag_{lag}"] = data[target_column].shift(lag)

        # Rolling statistics για CGM
        if target_column in data.columns:
            windows = [6, 12, 24]  # 30, 60, 120 λεπτά
            for window in windows:
                data[f"{target_column}_mean_{window}"] = data[target_column].rolling(window).mean()
                data[f"{target_column}_std_{window}"] = data[target_column].rolling(window).std()
                data[f"{target_column}_slope_{window}"] = self._calculate_slope(
                    data[target_column], window
                )

        # Χρόνος από τελευταίο γεύμα
        if "carbs" in data.columns:
            data["time_since_meal"] = self._time_since_event(data["carbs"] > 0)
            data["carbs_cumulative_4h"] = data["carbs"].rolling("4h").sum()

        # Χρόνος από τελευταίο bolus
        if "bolus" in data.columns:
            data["time_since_bolus"] = self._time_since_event(data["bolus"] > 0)
            data["bolus_cumulative_2h"] = data["bolus"].rolling("2h").sum()

        # Ενεργός ινσουλίνη (απλοποιημένο μοντέλο)
        if "bolus" in data.columns:
            data["active_insulin"] = self._calculate_active_insulin(data["bolus"])

        # Χρονικά χαρακτηριστικά
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

        # Κυκλικά χαρακτηριστικά για ώρα
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

        return data

    def normalize_features(
        self, data: pd.DataFrame, fit: bool = True, method: str = "standard"
    ) -> pd.DataFrame:
        """
        Κανονικοποίηση χαρακτηριστικών.

        Args:
            data: Δεδομένα προς κανονικοποίηση
            fit: Αν True, fit τους scalers
            method: Μέθοδος κανονικοποίησης ('standard', 'minmax')

        Returns:
            pd.DataFrame: Κανονικοποιημένα δεδομένα
        """
        data = data.copy()

        # Επιλογή στηλών προς κανονικοποίηση
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ["hour", "day_of_week", "is_weekend"]  # Κατηγορικά χαρακτηριστικά
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

        if fit:
            self.feature_columns = cols_to_scale

        for col in cols_to_scale:
            if col not in data.columns:
                continue

            if fit:
                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Μη υποστηριζόμενη μέθοδος: {method}")

                # Fit scaler στα μη-NaN δεδομένα
                valid_data = data[col].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    scaler.fit(valid_data)
                    self.scalers[col] = scaler

            # Transform
            if col in self.scalers:
                valid_mask = ~data[col].isna()
                if valid_mask.any():
                    data.loc[valid_mask, col] = (
                        self.scalers[col]
                        .transform(data.loc[valid_mask, col].values.reshape(-1, 1))
                        .flatten()
                    )

        return data

    def create_sequences(
        self,
        data: pd.DataFrame,
        target_column: str = "cgm",
        sequence_length: int = 12,
        prediction_horizon: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Δημιουργία sequences για RNN/LSTM μοντέλα.

        Args:
            data: Προεπεξεργασμένα δεδομένα
            target_column: Στήλη target
            sequence_length: Μήκος ιστορικής ακολουθίας
            prediction_horizon: Ορίζοντας πρόβλεψης

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_sequences, y_sequences)
        """
        if target_column not in data.columns:
            raise ValueError(f"Στήλη '{target_column}' δεν βρέθηκε στα δεδομένα")

        # Αφαίρεση rows με NaN στο target
        clean_data = data.dropna(subset=[target_column])

        if len(clean_data) < sequence_length + prediction_horizon:
            raise ValueError("Δεν υπάρχουν αρκετά δεδομένα για sequences")

        # Επιλογή feature columns (εκτός από target)
        feature_cols = [col for col in clean_data.columns if col != target_column]
        X_data = clean_data[feature_cols].values
        y_data = clean_data[target_column].values

        X_sequences = []
        y_sequences = []

        for i in range(len(clean_data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            X_seq = X_data[i : i + sequence_length]

            # Target sequence (μπορεί να είναι ένα ή περισσότερα steps)
            y_seq = y_data[i + sequence_length : i + sequence_length + prediction_horizon]

            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

        return np.array(X_sequences), np.array(y_sequences)

    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Υπολογισμός slope σε κυλιόμενο παράθυρο."""

        def calc_slope(values):
            if len(values) < 2 or values.isna().all():
                return np.nan
            x = np.arange(len(values))
            valid_mask = ~values.isna()
            if valid_mask.sum() < 2:
                return np.nan
            return np.polyfit(x[valid_mask], values[valid_mask], 1)[0]

        return series.rolling(window).apply(calc_slope, raw=False)

    def _time_since_event(self, event_mask: pd.Series) -> pd.Series:
        """Υπολογισμός χρόνου από τελευταίο event."""
        time_since = pd.Series(index=event_mask.index, dtype=float)
        last_event_time = None

        for timestamp, is_event in event_mask.items():
            if is_event:
                last_event_time = timestamp
                time_since[timestamp] = 0
            elif last_event_time is not None:
                time_since[timestamp] = (
                    timestamp - last_event_time
                ).total_seconds() / 60  # σε λεπτά
            else:
                time_since[timestamp] = np.inf

        return time_since

    def _calculate_active_insulin(self, bolus_series: pd.Series) -> pd.Series:
        """
        Απλοποιημένος υπολογισμός ενεργού ινσουλίνης (IOB).
        Χρησιμοποιεί εκθετική φθίνουσα καμπύλη με DIA=3h.
        """
        dia_minutes = 180  # Duration of Insulin Action σε λεπτά
        decay_rate = np.log(2) / dia_minutes  # Half-life based decay

        active_insulin = pd.Series(index=bolus_series.index, dtype=float)
        iob = 0

        for i, (timestamp, bolus) in enumerate(bolus_series.items()):
            if i > 0:
                # Χρονικό διάστημα από προηγούμενη μέτρηση
                time_diff = (timestamp - bolus_series.index[i - 1]).total_seconds() / 60
                # Φθίνουσα ενεργός ινσουλίνη
                iob *= np.exp(-decay_rate * time_diff)

            # Προσθήκη νέου bolus
            if bolus > 0:
                iob += bolus

            active_insulin[timestamp] = iob

        return active_insulin

    def get_feature_names(self) -> Optional[List[str]]:
        """Επιστροφή ονομάτων χαρακτηριστικών."""
        return self.feature_columns

    def inverse_transform_target(
        self, values: np.ndarray, target_column: str = "cgm"
    ) -> np.ndarray:
        """Αντίστροφη κανονικοποίηση για τιμές target."""
        if target_column in self.scalers:
            return self.scalers[target_column].inverse_transform(values.reshape(-1, 1)).flatten()
        return values


def create_train_test_split(
    data: pd.DataFrame, test_size: float = 0.2, method: str = "temporal"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Δημιουργία train/test split για χρονοσειρές.

    Args:
        data: Δεδομένα προς διαχωρισμό
        test_size: Ποσοστό δεδομένων για test
        method: Μέθοδος διαχωρισμού ('temporal', 'random')

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
    """
    if method == "temporal":
        # Χρονικός διαχωρισμός (τα τελευταία δεδομένα για test)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

    elif method == "random":
        # Τυχαίος διαχωρισμός (προσοχή με χρονοσειρές!)
        test_indices = np.random.choice(len(data), size=int(len(data) * test_size), replace=False)
        train_indices = np.setdiff1d(np.arange(len(data)), test_indices)
        train_data = data.iloc[train_indices]
        test_data = data.iloc[test_indices]

    else:
        raise ValueError(f"Μη υποστηριζόμενη μέθοδος: {method}")

    return train_data, test_data


def validate_data_quality(
    data: pd.DataFrame, target_column: str = "cgm", min_data_points: int = 100
) -> Dict[str, Union[bool, float, str]]:
    """
    Έλεγχος ποιότητας δεδομένων.

    Args:
        data: Δεδομένα προς έλεγχο
        target_column: Στήλη target
        min_data_points: Ελάχιστος αριθμός σημείων

    Returns:
        Dict: Αποτελέσματα ελέγχου ποιότητας
    """
    results = {
        "sufficient_data": len(data) >= min_data_points,
        "target_coverage": 0.0,
        "missing_data_percentage": 0.0,
        "data_frequency_consistent": True,
        "outlier_percentage": 0.0,
        "quality_score": 0.0,
        "recommendations": [],
    }

    if target_column in data.columns:
        # Coverage του target
        results["target_coverage"] = (~data[target_column].isna()).mean()

        # Ποσοστό ελλειπόντων δεδομένων
        results["missing_data_percentage"] = data.isna().mean().mean()

        # Έλεγχος outliers στο target
        if results["target_coverage"] > 0:
            q1 = data[target_column].quantile(0.25)
            q3 = data[target_column].quantile(0.75)
            iqr = q3 - q1
            outliers = (
                (data[target_column] < q1 - 1.5 * iqr) | (data[target_column] > q3 + 1.5 * iqr)
            ).sum()
            results["outlier_percentage"] = outliers / len(data)

    # Συνολικό σκορ ποιότητας
    quality_factors = [
        results["target_coverage"],
        1 - results["missing_data_percentage"],
        1 - results["outlier_percentage"],
        float(results["sufficient_data"]),
    ]
    results["quality_score"] = np.mean(quality_factors)

    # Συστάσεις
    if results["target_coverage"] < 0.8:
        results["recommendations"].append("Χαμηλό coverage CGM δεδομένων")
    if results["missing_data_percentage"] > 0.3:
        results["recommendations"].append("Πολλά ελλείποντα δεδομένα")
    if results["outlier_percentage"] > 0.1:
        results["recommendations"].append("Πολλά outliers - εξετάστε καθαρισμό")
    if not results["sufficient_data"]:
        results["recommendations"].append(
            f"Λίγα δεδομένα - χρειάζονται τουλάχιστον {min_data_points}"
        )

    return results
