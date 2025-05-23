"""
Baseline μοντέλα για πρόβλεψη γλυκόζης.

Περιλαμβάνει:
- ARIMA: Αυτοπαλίνδρομο ολοκληρωμένο μοντέλο κινητού μέσου
- Prophet: Facebook's προγνωστικό μοντέλο για χρονοσειρές
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.twin import BaseModel

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ARIMAModel(BaseModel):
    """
    ARIMA μοντέλο για πρόβλεψη γλυκόζης.
    
    Παράδειγμα χρήσης:
    >>> model = ARIMAModel(order=(1, 1, 1))
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test, horizon=60)
    """
    
    def __init__(self,
                 order: tuple = (1, 1, 1),
                 seasonal_order: tuple = (0, 0, 0, 0),
                 auto_arima: bool = True,
                 max_p: int = 5,
                 max_d: int = 2,
                 max_q: int = 5):
        """
        Αρχικοποίηση ARIMA μοντέλου.
        
        Args:
            order: (p, d, q) παράμετροι ARIMA
            seasonal_order: (P, D, Q, s) εποχιακές παράμετροι
            auto_arima: Αν True, αυτόματη επιλογή παραμέτρων
            max_p, max_d, max_q: Μέγιστες τιμές για auto_arima
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Το statsmodels είναι απαραίτητο για ARIMA. pip install statsmodels")
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.feature_names = None
        self.best_order = None
        
    def _find_best_order(self, y: pd.Series) -> tuple:
        """Εύρεση βέλτιστων παραμέτρων με grid search."""
        from itertools import product
        
        best_aic = float('inf')
        best_order = self.order
        
        # Grid search για p, d, q
        p_values = range(0, self.max_p + 1)
        d_values = range(0, self.max_d + 1)
        q_values = range(0, self.max_q + 1)
        
        for p, d, q in product(p_values, d_values, q_values):
            try:
                model = ARIMA(y, order=(p, d, q))
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    
            except Exception:
                continue
        
        return best_order
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ARIMAModel':
        """
        Εκπαίδευση του ARIMA μοντέλου.
        
        Args:
            X: Features DataFrame (θα αγνοηθεί, το ARIMA χρησιμοποιεί μόνο y)
            y: Target Series (CGM values)
            **kwargs: Επιπλέον παράμετροι
        """
        self.feature_names = X.columns.tolist() if X is not None else None
        
        # Έλεγχος για missing values
        if y.isnull().any():
            y = y.interpolate().fillna(method='bfill').fillna(method='ffill')
        
        # Αυτόματη επιλογή παραμέτρων αν ζητείται
        if self.auto_arima:
            print("Αναζήτηση βέλτιστων παραμέτρων ARIMA...")
            self.best_order = self._find_best_order(y)
            print(f"Βέλτιστες παράμετροι: {self.best_order}")
        else:
            self.best_order = self.order
        
        # Δημιουργία και εκπαίδευση μοντέλου
        self.model = ARIMA(y, order=self.best_order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit()
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame = None, horizon: int = 30, **kwargs) -> np.ndarray:
        """
        Πρόβλεψη γλυκόζης.
        
        Args:
            X: Features DataFrame (αγνοείται)
            horizon: Ορίζοντας πρόβλεψης σε λεπτά
            **kwargs: Επιπλέον παράμετροι
            
        Returns:
            Προβλέψεις γλυκόζης
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")
        
        # Μετατροπή λεπτών σε time steps (υποθέτουμε 5-min intervals)
        steps = max(1, horizon // 5)
        
        # Πρόβλεψη
        forecast = self.fitted_model.forecast(steps=steps)
        
        return forecast.values if hasattr(forecast, 'values') else forecast
    
    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του μοντέλου."""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'auto_arima': self.auto_arima,
            'max_p': self.max_p,
            'max_d': self.max_d,
            'max_q': self.max_q,
            'best_order': self.best_order
        }
    
    def set_params(self, **params) -> 'ARIMAModel':
        """Ορισμός παραμέτρων του μοντέλου."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Άγνωστη παράμετρος: {key}")
        
        # Επαναρχικοποίηση
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
        return self


class ProphetModel(BaseModel):
    """
    Prophet μοντέλο για πρόβλεψη γλυκόζης.
    
    Παράδειγμα χρήσης:
    >>> model = ProphetModel(seasonality_mode='additive')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test, horizon=60)
    """
    
    def __init__(self,
                 growth: str = 'linear',
                 seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = False,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = True,
                 interval_width: float = 0.8,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        """
        Αρχικοποίηση Prophet μοντέλου.
        
        Args:
            growth: Τύπος ανάπτυξης ('linear' ή 'logistic')
            seasonality_mode: Τρόπος εποχιακότητας ('additive' ή 'multiplicative')
            yearly_seasonality: Ετήσια εποχιακότητα
            weekly_seasonality: Εβδομαδιαία εποχιακότητα
            daily_seasonality: Ημερήσια εποχιακότητα
            interval_width: Εύρος confidence intervals
            changepoint_prior_scale: Ευελιξία αλλαγών τάσης
            seasonality_prior_scale: Ευελιξία εποχιακότητας
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Το Prophet είναι απαραίτητο. pip install prophet")
        
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.last_date = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ProphetModel':
        """
        Εκπαίδευση του Prophet μοντέλου.
        
        Args:
            X: Features DataFrame
            y: Target Series (CGM values)
            **kwargs: Επιπλέον παράμετροι
        """
        self.feature_names = X.columns.tolist() if X is not None else None
        
        # Προετοιμασία δεδομένων για Prophet
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
        
        # Διασφάλιση ότι το ds είναι datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        
        # Αποθήκευση τελευταίας ημερομηνίας
        self.last_date = df['ds'].max()
        
        # Δημιουργία Prophet μοντέλου
        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )
        
        # Προσθήκη custom εποχιακότητας για ωριαίο pattern (διαβήτης)
        self.model.add_seasonality(name='hourly', period=1, fourier_order=5)
        
        # Εκπαίδευση
        self.model.fit(df)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame = None, horizon: int = 30, **kwargs) -> np.ndarray:
        """
        Πρόβλεψη γλυκόζης.
        
        Args:
            X: Features DataFrame (αγνοείται)
            horizon: Ορίζοντας πρόβλεψης σε λεπτά
            **kwargs: Επιπλέον παράμετροι
            
        Returns:
            Προβλέψεις γλυκόζης
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")
        
        # Δημιουργία μελλοντικών ημερομηνιών
        periods = max(1, horizon // 5)  # Υποθέτουμε 5-min intervals
        
        future = self.model.make_future_dataframe(periods=periods, freq='5T')
        
        # Πρόβλεψη
        forecast = self.model.predict(future)
        
        # Επιστροφή μόνο των μελλοντικών προβλέψεων
        future_predictions = forecast['yhat'].tail(periods)
        
        return future_predictions.values
    
    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του μοντέλου."""
        return {
            'growth': self.growth,
            'seasonality_mode': self.seasonality_mode,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'interval_width': self.interval_width,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale
        }
    
    def set_params(self, **params) -> 'ProphetModel':
        """Ορισμός παραμέτρων του μοντέλου."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Άγνωστη παράμετρος: {key}")
        
        # Επαναρχικοποίηση
        self.model = None
        self.is_fitted = False
        
        return self
    
    def get_forecast_components(self) -> Optional[pd.DataFrame]:
        """Επιστροφή των συστατικών της πρόβλεψης (trend, seasonality κλπ)."""
        if not self.is_fitted:
            return None
        
        # Πρόβλεψη για την τελευταία περίοδο
        future = self.model.make_future_dataframe(periods=1)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'trend', 'seasonal', 'weekly', 'daily']].tail(10) 