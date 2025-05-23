"""
Μετρικές αξιολόγησης για ψηφιακό δίδυμο διαβήτη.

Περιλαμβάνει στατιστικές και κλινικές μετρικές:
- Στατιστικές: RMSE, MAE, MAPE, MASE
- Κλινικές: Clarke Error Grid, Parkes Error Grid, Time in Range
- Γλυκαιμικός κίνδυνος: LBGI, HBGI, LBGI/HBGI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred: Προβλέψεις
        
    Returns:
        RMSE τιμή
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred: Προβλέψεις
        
    Returns:
        MAE τιμή
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred: Προβλέψεις
        
    Returns:
        MAPE τιμή σε ποσοστό
    """
    # Αποφυγή διαίρεσης με μηδέν
    mask = y_true != 0
    if not mask.any():
        return float('inf')
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None) -> float:
    """
    Mean Absolute Scaled Error.
    
    Args:
        y_true: Πραγματικές τιμές test
        y_pred: Προβλέψεις
        y_train: Training τιμές για υπολογισμό baseline
        
    Returns:
        MASE τιμή
    """
    if y_train is None:
        # Χρησιμοποιούμε naive forecast (τελευταία τιμή) ως baseline
        naive_forecast = np.concatenate([[y_true[0]], y_true[:-1]])
        mae_naive = mae(y_true, naive_forecast)
    else:
        # Χρησιμοποιούμε seasonal naive forecast
        naive_forecast = np.concatenate([[y_train[-1]], y_train[:-1]])
        mae_naive = mae(y_train, naive_forecast)
    
    if mae_naive == 0:
        return float('inf')
    
    return mae(y_true, y_pred) / mae_naive


def clarke_error_grid(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Clarke Error Grid Analysis.
    
    Κατηγοριοποιεί τις προβλέψεις σε ζώνες κλινικής ακρίβειας:
    - Zone A: Κλινικά ακριβείς
    - Zone B: Αποδεκτές
    - Zone C: Υπερβολικές διορθώσεις
    - Zone D: Δυνητικά επικίνδυνες
    - Zone E: Εσφαλμένες
    
    Args:
        y_true: Πραγματικές τιμές γλυκόζης (mg/dL)
        y_pred: Προβλεπόμενες τιμές γλυκόζης (mg/dL)
        
    Returns:
        Dictionary με ποσοστά για κάθε ζώνη
    """
    # Μετατροπή σε numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Αρχικοποίηση ζωνών
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    total = len(y_true)
    
    for i in range(total):
        ref_value = y_true[i]
        pred_value = y_pred[i]
        
        # Zone A - Κλινικά ακριβείς
        if (ref_value <= 70 and pred_value <= 70) or \
           (pred_value <= 1.2 * ref_value and pred_value >= 0.8 * ref_value):
            zones['A'] += 1
            
        # Zone B - Αποδεκτές
        elif (ref_value >= 180 and pred_value <= 70) or \
             (ref_value <= 70 and pred_value >= 180):
            zones['E'] += 1  # Αυτό είναι στην πραγματικότητα Zone E
            
        elif ref_value >= 70 and ref_value <= 290:
            if pred_value >= ref_value + 110:
                zones['C'] += 1
            elif pred_value <= ref_value - 110:
                zones['C'] += 1
            elif (ref_value <= 70 and pred_value <= 180 and pred_value > 70) or \
                 (ref_value >= 180 and pred_value > 70 and pred_value <= 180):
                zones['D'] += 1
            else:
                zones['B'] += 1
        else:
            zones['B'] += 1
    
    # Μετατροπή σε ποσοστά
    zone_percentages = {zone: (count / total) * 100 for zone, count in zones.items()}
    
    return zone_percentages


def parkes_error_grid(y_true: np.ndarray, y_pred: np.ndarray, 
                     diabetes_type: str = 'type1') -> Dict[str, float]:
    """
    Parkes Error Grid Analysis.
    
    Παρόμοια με Clarke Grid αλλά με διαφορετικά όρια για διαβήτη τύπου 1 και 2.
    
    Args:
        y_true: Πραγματικές τιμές γλυκόζης (mg/dL)
        y_pred: Προβλεπόμενες τιμές γλυκόζης (mg/dL)
        diabetes_type: 'type1' ή 'type2'
        
    Returns:
        Dictionary με ποσοστά για κάθε ζώνη
    """
    # Απλουστευμένη υλοποίηση - χρησιμοποιεί Clarke Grid
    # Στην πραγματικότητα, το Parkes Grid έχει διαφορετικά όρια
    zones = clarke_error_grid(y_true, y_pred)
    
    # TODO: Υλοποίηση πραγματικού Parkes Grid
    return zones


def time_in_range(glucose_values: np.ndarray, 
                  target_range: Tuple[float, float] = (70, 180),
                  tight_range: Tuple[float, float] = (70, 140)) -> Dict[str, float]:
    """
    Time in Range (TIR) ανάλυση.
    
    Args:
        glucose_values: Τιμές γλυκόζης (mg/dL)
        target_range: Στόχος εύρος (default: 70-180 mg/dL)
        tight_range: Στενό εύρος (default: 70-140 mg/dL)
        
    Returns:
        Dictionary με ποσοστά χρόνου σε διάφορα εύρη
    """
    glucose_values = np.asarray(glucose_values)
    total_readings = len(glucose_values)
    
    # Κατηγορίες γλυκόζης
    very_low = np.sum(glucose_values < 54)      # <54 mg/dL
    low = np.sum((glucose_values >= 54) & (glucose_values < 70))  # 54-70 mg/dL
    in_range = np.sum((glucose_values >= target_range[0]) & 
                     (glucose_values <= target_range[1]))  # 70-180 mg/dL
    in_tight_range = np.sum((glucose_values >= tight_range[0]) & 
                           (glucose_values <= tight_range[1]))  # 70-140 mg/dL
    high = np.sum((glucose_values > 180) & (glucose_values <= 250))  # 180-250 mg/dL
    very_high = np.sum(glucose_values > 250)    # >250 mg/dL
    
    return {
        'very_low_percent': (very_low / total_readings) * 100,
        'low_percent': (low / total_readings) * 100,
        'in_range_percent': (in_range / total_readings) * 100,
        'in_tight_range_percent': (in_tight_range / total_readings) * 100,
        'high_percent': (high / total_readings) * 100,
        'very_high_percent': (very_high / total_readings) * 100,
        'below_range_percent': ((very_low + low) / total_readings) * 100,
        'above_range_percent': ((high + very_high) / total_readings) * 100
    }


def glycemic_risk_index(glucose_values: np.ndarray) -> Dict[str, float]:
    """
    Υπολογισμός γλυκαιμικών δεικτών κινδύνου.
    
    Περιλαμβάνει:
    - LBGI: Low Blood Glucose Index
    - HBGI: High Blood Glucose Index
    - ADRR: Average Daily Risk Range
    
    Args:
        glucose_values: Τιμές γλυκόζης (mg/dL)
        
    Returns:
        Dictionary με δείκτες κινδύνου
    """
    glucose_values = np.asarray(glucose_values)
    
    # Μετατροπή σε λογαριθμική κλίμακα
    # f(BG) = 1.509 * (log(BG)^1.084 - 5.381)
    log_glucose = np.log(glucose_values)
    f_bg = 1.509 * (log_glucose ** 1.084 - 5.381)
    
    # Υπολογισμός risk function
    # r(BG) = 10 * f(BG)^2
    risk_bg = 10 * (f_bg ** 2)
    
    # LBGI - Low Blood Glucose Index
    low_risk = np.where(f_bg < 0, risk_bg, 0)
    lbgi = np.mean(low_risk)
    
    # HBGI - High Blood Glucose Index  
    high_risk = np.where(f_bg > 0, risk_bg, 0)
    hbgi = np.mean(high_risk)
    
    # Συνολικός δείκτης κινδύνου
    total_risk = lbgi + hbgi
    
    # Κατηγοριοποίηση κινδύνου
    if lbgi < 1.1:
        lbgi_risk = 'minimal'
    elif lbgi < 2.5:
        lbgi_risk = 'low'
    elif lbgi < 5.0:
        lbgi_risk = 'moderate'
    else:
        lbgi_risk = 'high'
    
    if hbgi < 4.5:
        hbgi_risk = 'minimal'
    elif hbgi < 9.0:
        hbgi_risk = 'low'
    elif hbgi < 15.0:
        hbgi_risk = 'moderate'
    else:
        hbgi_risk = 'high'
    
    return {
        'lbgi': lbgi,
        'hbgi': hbgi,
        'total_risk': total_risk,
        'lbgi_risk_level': lbgi_risk,
        'hbgi_risk_level': hbgi_risk
    }


def coefficient_of_variation(glucose_values: np.ndarray) -> float:
    """
    Συντελεστής μεταβλητότητας γλυκόζης.
    
    Args:
        glucose_values: Τιμές γλυκόζης
        
    Returns:
        CV ποσοστό
    """
    return (np.std(glucose_values) / np.mean(glucose_values)) * 100


def glucose_management_indicator(glucose_values: np.ndarray) -> float:
    """
    Glucose Management Indicator (GMI).
    
    Εκτιμά την HbA1c βάσει CGM δεδομένων.
    
    Args:
        glucose_values: Τιμές γλυκόζης (mg/dL)
        
    Returns:
        GMI τιμή (%)
    """
    mean_glucose = np.mean(glucose_values)
    # GMI (%) = 3.31 + 0.02392 × mean glucose (mg/dL)
    gmi = 3.31 + 0.02392 * mean_glucose
    return gmi


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     metrics: Optional[List[str]] = None,
                     **kwargs) -> Dict[str, Union[float, Dict]]:
    """
    Υπολογισμός όλων των μετρικών.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred: Προβλέψεις
        metrics: Λίστα μετρικών προς υπολογισμό
        **kwargs: Επιπλέον παράμετροι
        
    Returns:
        Dictionary με όλες τις μετρικές
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'mape', 'clarke_grid', 'time_in_range']
    
    results = {}
    
    # Στατιστικές μετρικές
    if 'rmse' in metrics:
        results['rmse'] = rmse(y_true, y_pred)
    
    if 'mae' in metrics:
        results['mae'] = mae(y_true, y_pred)
    
    if 'mape' in metrics:
        results['mape'] = mape(y_true, y_pred)
    
    if 'mase' in metrics:
        y_train = kwargs.get('y_train', None)
        results['mase'] = mase(y_true, y_pred, y_train)
    
    # Κλινικές μετρικές
    if 'clarke_grid' in metrics:
        results['clarke_grid'] = clarke_error_grid(y_true, y_pred)
    
    if 'parkes_grid' in metrics:
        diabetes_type = kwargs.get('diabetes_type', 'type1')
        results['parkes_grid'] = parkes_error_grid(y_true, y_pred, diabetes_type)
    
    if 'time_in_range' in metrics:
        # Χρησιμοποιούμε τις πραγματικές τιμές για TIR
        results['time_in_range'] = time_in_range(y_true)
    
    if 'time_in_range_pred' in metrics:
        # TIR για προβλέψεις
        results['time_in_range_pred'] = time_in_range(y_pred)
    
    if 'glycemic_risk' in metrics:
        results['glycemic_risk_true'] = glycemic_risk_index(y_true)
        results['glycemic_risk_pred'] = glycemic_risk_index(y_pred)
    
    if 'cv' in metrics:
        results['cv_true'] = coefficient_of_variation(y_true)
        results['cv_pred'] = coefficient_of_variation(y_pred)
    
    if 'gmi' in metrics:
        results['gmi_true'] = glucose_management_indicator(y_true)
        results['gmi_pred'] = glucose_management_indicator(y_pred)
    
    return results


def prediction_interval_coverage(y_true: np.ndarray, 
                                y_pred_lower: np.ndarray,
                                y_pred_upper: np.ndarray,
                                confidence_level: float = 0.95) -> float:
    """
    Κάλυψη διαστημάτων εμπιστοσύνης.
    
    Args:
        y_true: Πραγματικές τιμές
        y_pred_lower: Κάτω όριο πρόβλεψης
        y_pred_upper: Άνω όριο πρόβλεψης
        confidence_level: Επίπεδο εμπιστοσύνης
        
    Returns:
        Ποσοστό κάλυψης
    """
    coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
    return coverage * 100 