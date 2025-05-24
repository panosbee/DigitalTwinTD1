"""
Model Evaluation Utilities για το Digital Twin SDK.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

def calculate_glucose_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray) -> Dict[str, float]:
    """
    Υπολογισμός glucose-specific evaluation metrics.
    
    Args:
        y_true: True glucose values
        y_pred: Predicted glucose values
        
    Returns:
        Dictionary με evaluation metrics
    """
    # Basic regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Glucose-specific metrics
    mard = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Time in range accuracy
    tir_true = ((y_true >= 70) & (y_true <= 180)).mean() * 100
    tir_pred = ((y_pred >= 70) & (y_pred <= 180)).mean() * 100
    tir_diff = abs(tir_true - tir_pred)
    
    # Hypoglycemia detection
    hypo_true = (y_true < 70).sum()
    hypo_pred = (y_pred < 70).sum()
    hypo_sensitivity = min(hypo_pred / max(hypo_true, 1), 1.0)
    
    return {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'r2': r2,
        'mard': mard,
        'time_in_range_accuracy': 100 - tir_diff,
        'hypoglycemia_sensitivity': hypo_sensitivity * 100
    }

def evaluate_prediction_horizon(y_true: np.ndarray,
                               predictions: List[np.ndarray],
                               horizons: List[int]) -> pd.DataFrame:
    """
    Αξιολόγηση accuracy σε διαφορετικούς prediction horizons.
    """
    results = []
    
    for i, (pred, horizon) in enumerate(zip(predictions, horizons)):
        if len(pred) == len(y_true):
            metrics = calculate_glucose_metrics(y_true, pred)
            metrics['horizon_minutes'] = horizon
            results.append(metrics)
    
    return pd.DataFrame(results)

def calculate_clinical_risk_score(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> Dict[str, float]:
    """
    Υπολογισμός clinical risk από prediction errors.
    """
    # Severe hypoglycemia missed (most dangerous)
    severe_hypo_true = y_true < 54
    severe_hypo_pred = y_pred < 54
    missed_severe_hypo = severe_hypo_true & ~severe_hypo_pred
    
    # Severe hyperglycemia missed
    severe_hyper_true = y_true > 250
    severe_hyper_pred = y_pred > 250
    missed_severe_hyper = severe_hyper_true & ~severe_hyper_pred
    
    # False alarms
    false_hypo_alarms = ~severe_hypo_true & severe_hypo_pred
    false_hyper_alarms = ~severe_hyper_true & severe_hyper_pred
    
    total_samples = len(y_true)
    
    return {
        'missed_severe_hypoglycemia_rate': missed_severe_hypo.sum() / total_samples * 100,
        'missed_severe_hyperglycemia_rate': missed_severe_hyper.sum() / total_samples * 100,
        'false_hypoglycemia_alarm_rate': false_hypo_alarms.sum() / total_samples * 100,
        'false_hyperglycemia_alarm_rate': false_hyper_alarms.sum() / total_samples * 100
    }

def assess_model_reliability(predictions: List[np.ndarray],
                           ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Αξιολόγηση reliability και consistency του model.
    """
    if len(predictions) < 2:
        return {'prediction_variance': 0.0, 'consistency_score': 100.0}
    
    # Calculate variance across multiple predictions
    pred_array = np.array(predictions)
    prediction_variance = np.mean(np.var(pred_array, axis=0))
    
    # Consistency score (lower variance = higher consistency)
    max_possible_variance = np.var(ground_truth)
    consistency_score = max(0, 100 * (1 - prediction_variance / max_possible_variance))
    
    return {
        'prediction_variance': prediction_variance,
        'consistency_score': consistency_score
    }

def time_lag_analysis(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     max_lag: int = 12) -> Dict[str, Union[int, float]]:
    """
    Ανάλυση time lag στις predictions.
    """
    best_correlation = -1
    best_lag = 0
    
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        elif lag > 0:
            if len(y_true) > lag:
                correlation = np.corrcoef(y_true[lag:], y_pred[:-lag])[0, 1]
            else:
                continue
        else:  # lag < 0
            if len(y_pred) > abs(lag):
                correlation = np.corrcoef(y_true[:lag], y_pred[abs(lag):])[0, 1]
            else:
                continue
                
        if not np.isnan(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    
    return {
        'optimal_lag_steps': best_lag,
        'optimal_correlation': best_correlation,
        'time_alignment_quality': min(100, best_correlation * 100)
    }

def comprehensive_evaluation(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            include_clinical: bool = True) -> Dict[str, float]:
    """
    Comprehensive evaluation που συνδυάζει όλα τα metrics.
    """
    results = {}
    
    # Basic glucose metrics
    results.update(calculate_glucose_metrics(y_true, y_pred))
    
    # Clinical risk assessment
    if include_clinical:
        results.update(calculate_clinical_risk_score(y_true, y_pred))
    
    # Time lag analysis
    results.update(time_lag_analysis(y_true, y_pred))
    
    # Overall clinical safety score
    safety_penalties = (
        results.get('missed_severe_hypoglycemia_rate', 0) * 3 +  # High penalty
        results.get('missed_severe_hyperglycemia_rate', 0) * 1 +
        results.get('false_hypoglycemia_alarm_rate', 0) * 0.5
    )
    
    results['clinical_safety_score'] = max(0, 100 - safety_penalties)
    
    return results

# Export for testing
__all__ = [
    'calculate_glucose_metrics',
    'evaluate_prediction_horizon',
    'calculate_clinical_risk_score',
    'assess_model_reliability',
    'time_lag_analysis',
    'comprehensive_evaluation'
] 