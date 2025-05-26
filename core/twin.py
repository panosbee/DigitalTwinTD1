"""
Κύρια κλάση DigitalTwin - high-level wrapper για όλα τα μοντέλα.
Παρέχει ενοποιημένη διεπαφή τύπου scikit-learn (fit/predict).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Βασική αφηρημένη κλάση για όλα τα μοντέλα του ψηφιακού διδύμου."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """Εκπαίδευση του μοντέλου."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, horizon: int = 30, **kwargs) -> np.ndarray:
        """Πρόβλεψη γλυκόζης για τον καθορισμένο ορίζοντα (σε λεπτά)."""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του μοντέλου."""
        pass

    @abstractmethod
    def set_params(self, **params) -> "BaseModel":
        """Ορισμός παραμέτρων του μοντέλου."""
        pass


class DigitalTwin:
    """
    Κύρια κλάση ψηφιακού διδύμου για διαβήτη τύπου 1.

    Παρέχει ενοποιημένη διεπαφή για:
    - Φόρτωση και προεπεξεργασία δεδομένων (CGM, γεύματα, ινσουλίνη, άσκηση)
    - Εκπαίδευση διαφορετικών τύπων μοντέλων (μηχανιστικά, LSTM, Transformer)
    - Πρόβλεψη γλυκόζης σε διάφορους ορίζοντες
    - Real-time simulation και αξιολόγηση

    Παράδειγμα χρήσης:
    >>> twin = DigitalTwin(model_type='lstm')
    >>> twin.fit(data)
    >>> predictions = twin.predict(horizon=60)  # 60 λεπτά μπροστά
    """

    def __init__(
        self,
        model_type: str = "lstm",
        model_params: Optional[Dict] = None,
        random_state: Optional[int] = None,
    ):
        """
        Αρχικοποίηση ψηφιακού διδύμου.

        Args:
            model_type: Τύπος μοντέλου ('lstm', 'transformer', 'mechanistic', 'arima', 'prophet')
            model_params: Παράμετροι για το μοντέλο
            random_state: Σπόρος τυχαιότητας για αναπαραγωγιμότητα
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.feature_columns = None
        self.target_column = "cgm"

        # Αρχικοποίηση μοντέλου βάσει τύπου
        self._initialize_model()

    def _initialize_model(self):
        """Αρχικοποίηση του επιλεγμένου μοντέλου."""
        import sys
        import os

        # Προσθήκη του root directory στο path
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_path not in sys.path:
            sys.path.append(root_path)

        if self.model_type == "lstm":
            from models.lstm import LSTMModel

            self.model = LSTMModel(**self.model_params)
        elif self.model_type == "transformer":
            from models.transformer import TransformerModel

            self.model = TransformerModel(**self.model_params)
        elif self.model_type == "mechanistic":
            from models.mechanistic import MechanisticModel

            self.model = MechanisticModel(**self.model_params)
        elif self.model_type == "arima":
            from models.baseline import ARIMAModel

            self.model = ARIMAModel(**self.model_params)
        elif self.model_type == "prophet":
            from models.baseline import ProphetModel

            self.model = ProphetModel(**self.model_params)
        else:
            raise ValueError(f"Μη υποστηριζόμενος τύπος μοντέλου: {self.model_type}")

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "cgm",
        feature_columns: Optional[List[str]] = None,
        validation_split: float = 0.2,
        **fit_kwargs,
    ) -> "DigitalTwin":
        """
        Εκπαίδευση του ψηφιακού διδύμου.

        Args:
            data: DataFrame με CGM, γεύματα, ινσουλίνη, κ.ά.
            target_column: Όνομα στήλης με τιμές γλυκόζης (default: 'cgm')
            feature_columns: Λίστα στηλών που θα χρησιμοποιηθούν ως features
            validation_split: Ποσοστό δεδομένων για validation
            **fit_kwargs: Επιπλέον παράμετροι για την εκπαίδευση

        Returns:
            self: Επιστρέφει τον εαυτό του για method chaining
        """
        if data.empty:
            raise ValueError("Τα δεδομένα είναι κενά")

        if target_column not in data.columns:
            raise ValueError(f"Η στήλη '{target_column}' δεν βρέθηκε στα δεδομένα")

        self.target_column = target_column

        # Αυτόματη επιλογή features αν δεν δίνονται
        if feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns

        # Προετοιμασία δεδομένων
        X = data[self.feature_columns]
        y = data[target_column]

        # Εκπαίδευση μοντέλου
        self.model.fit(X, y, **fit_kwargs)
        self.is_fitted = True

        return self

    def predict(
        self,
        X: Optional[pd.DataFrame] = None,
        horizon: int = 30,
        confidence_interval: float = 0.95,
        return_confidence: bool = False,
        **predict_kwargs,
    ) -> Union[np.ndarray, tuple]:
        """
        Πρόβλεψη γλυκόζης.

        Args:
            X: Δεδομένα εισόδου (αν None, χρησιμοποιεί τελευταία γνωστά)
            horizon: Ορίζοντας πρόβλεψης σε λεπτά (default: 30)
            confidence_interval: Επίπεδο εμπιστοσύνης για intervals
            return_confidence: Αν True, επιστρέφει και confidence intervals
            **predict_kwargs: Επιπλέον παράμετροι

        Returns:
            np.ndarray ή tuple: Προβλέψεις (και confidence intervals αν ζητηθούν)
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Καλέστε fit() πρώτα.")

        predictions = self.model.predict(X, horizon=horizon, **predict_kwargs)

        if return_confidence:
            # Υπολογισμός confidence intervals (απλοποιημένος)
            std = np.std(predictions) if len(predictions) > 1 else 0.1
            alpha = 1 - confidence_interval
            margin = 1.96 * std  # Για 95% CI

            lower_bound = predictions - margin
            upper_bound = predictions + margin

            return predictions, (lower_bound, upper_bound)

        return predictions

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Αξιολόγηση του μοντέλου.

        Args:
            X_test: Test features
            y_test: Test targets
            metrics: Λίστα μετρικών ['rmse', 'mae', 'mape', 'clarke_a', 'clarke_b']

        Returns:
            Dict με τα αποτελέσματα των μετρικών
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί.")

        if metrics is None:
            metrics = ["rmse", "mae", "mape"]

        predictions = self.predict(X_test)

        # Προσθήκη του root directory στο path για το import
        import sys
        import os

        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_path not in sys.path:
            sys.path.append(root_path)

        from utils.metrics import calculate_metrics

        return calculate_metrics(y_test, predictions, metrics)

    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του ψηφιακού διδύμου."""
        return {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "random_state": self.random_state,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
        }

    def set_params(self, **params) -> "DigitalTwin":
        """Ορισμός παραμέτρων του ψηφιακού διδύμου."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Άγνωστη παράμετρος: {key}")

        # Επαναρχικοποίηση μοντέλου αν άλλαξε ο τύπος
        if "model_type" in params or "model_params" in params:
            self._initialize_model()
            self.is_fitted = False

        return self

    def save_model(self, filepath: str):
        """Αποθήκευση του εκπαιδευμένου μοντέλου."""
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί.")

        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filepath: str) -> "DigitalTwin":
        """Φόρτωση αποθηκευμένου μοντέλου."""
        import pickle

        with open(filepath, "rb") as f:
            return pickle.load(f)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Επιστροφή σημαντικότητας features (αν υποστηρίζεται από το μοντέλο)."""
        if hasattr(self.model, "get_feature_importance"):
            return self.model.get_feature_importance()
        return None

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"DigitalTwin(model_type='{self.model_type}', status='{status}')"
