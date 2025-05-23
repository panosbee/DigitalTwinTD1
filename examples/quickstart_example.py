"""
Παράδειγμα γρήγορης εκκίνησης - Ψηφιακός Δίδυμος για Διαβήτη Τύπου 1

Αυτό το παράδειγμα δείχνει:
1. Πώς να φορτώσετε δεδομένα CGM
2. Πώς να εκπαιδεύσετε διαφορετικά μοντέλα
3. Πώς να κάνετε προβλέψεις
4. Πώς να αξιολογήσετε τα αποτελέσματα
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports της βιβλιοθήκης
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.twin import DigitalTwin
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.baseline import ARIMAModel, ProphetModel
from models.mechanistic import MechanisticModel
from utils.metrics import calculate_metrics
from utils.visualization import plot_predictions, plot_metrics_comparison


def generate_synthetic_data(days: int = 7, noise_level: float = 0.1):
    """
    Δημιουργία συνθετικών δεδομένων για το παράδειγμα.
    
    Args:
        days: Αριθμός ημερών
        noise_level: Επίπεδο θορύβου
        
    Returns:
        DataFrame με συνθετικά δεδομένα
    """
    # Χρονικό εύρος (5-min intervals)
    start_time = datetime.now() - timedelta(days=days)
    time_points = pd.date_range(start_time, periods=days * 288, freq='5T')
    
    # Βασική τάση γλυκόζης (circadian rhythm)
    base_glucose = 100 + 20 * np.sin(2 * np.pi * np.arange(len(time_points)) / 288)
    
    # Προσθήκη τάσης για γεύματα (3 φορές την ημέρα)
    meal_times = []
    glucose_with_meals = base_glucose.copy()
    carbs = np.zeros(len(time_points))
    insulin = np.zeros(len(time_points))
    
    for day in range(days):
        # Πρωινό (08:00), Μεσημεριανό (13:00), Βραδινό (19:00)
        for meal_hour in [8, 13, 19]:
            meal_idx = day * 288 + meal_hour * 12  # 12 intervals per hour
            if meal_idx < len(time_points):
                # Γεύμα: αύξηση γλυκόζης
                peak_time = meal_idx + 24  # Peak σε 2 ώρες
                if peak_time < len(glucose_with_meals):
                    meal_carbs = np.random.normal(50, 15)  # 30-70g υδατάνθρακες
                    carbs[meal_idx] = max(0, meal_carbs)
                    
                    # Gaussian peak για γεύμα
                    meal_effect = 80 * np.exp(-((np.arange(len(glucose_with_meals)) - peak_time) ** 2) / (2 * 20 ** 2))
                    glucose_with_meals += meal_effect
                    
                    # Ινσουλίνη: bolus στο γεύμα
                    insulin_dose = meal_carbs / 10  # 1U per 10g carbs
                    insulin[meal_idx] = max(0, insulin_dose)
                    
                    # Ινσουλίνη effect (μείωση γλυκόζης)
                    insulin_effect = -60 * np.exp(-((np.arange(len(glucose_with_meals)) - meal_idx - 12) ** 2) / (2 * 15 ** 2))
                    glucose_with_meals += insulin_effect
    
    # Προσθήκη θορύβου
    noise = np.random.normal(0, noise_level * np.std(glucose_with_meals), len(glucose_with_meals))
    final_glucose = glucose_with_meals + noise
    
    # Διασφάλιση ρεαλιστικών τιμών
    final_glucose = np.clip(final_glucose, 50, 400)
    
    # Δημιουργία DataFrame
    data = pd.DataFrame({
        'timestamp': time_points,
        'cgm': final_glucose,
        'carbs': carbs,
        'insulin': insulin,
        'activity': np.random.exponential(0.1, len(time_points)),  # Χαμηλή δραστηριότητα
        'stress': np.random.beta(2, 5, len(time_points)),  # Στρες factor
    })
    
    data.set_index('timestamp', inplace=True)
    return data


def prepare_features(data: pd.DataFrame, lookback: int = 12):
    """
    Προετοιμασία features για τα μοντέλα.
    
    Args:
        data: Τα βασικά δεδομένα
        lookback: Αριθμός προηγούμενων τιμών που θα χρησιμοποιηθούν
        
    Returns:
        DataFrame με features
    """
    features_df = data.copy()
    
    # Lag features για γλυκόζη
    for i in range(1, lookback + 1):
        features_df[f'cgm_lag_{i}'] = features_df['cgm'].shift(i)
    
    # Moving averages
    features_df['cgm_ma_3'] = features_df['cgm'].rolling(window=3).mean()
    features_df['cgm_ma_6'] = features_df['cgm'].rolling(window=6).mean()
    features_df['cgm_ma_12'] = features_df['cgm'].rolling(window=12).mean()
    
    # Διαφορές
    features_df['cgm_diff_1'] = features_df['cgm'].diff()
    features_df['cgm_diff_2'] = features_df['cgm'].diff(2)
    
    # Ώρα της ημέρας (cyclic encoding)
    features_df['hour'] = features_df.index.hour
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    
    # Μέρα της εβδομάδας
    features_df['day_of_week'] = features_df.index.dayofweek
    features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
    
    # Cumulative carbs και insulin (τελευταίες 4 ώρες)
    features_df['carbs_4h'] = features_df['carbs'].rolling(window=48).sum()
    features_df['insulin_4h'] = features_df['insulin'].rolling(window=48).sum()
    
    # Αφαίρεση στηλών με πολλά NaN
    features_df = features_df.dropna()
    
    return features_df


def main():
    """Κύρια συνάρτηση παραδείγματος."""
    
    print("🔬 Παράδειγμα Ψηφιακού Διδύμου για Διαβήτη Τύπου 1")
    print("=" * 55)
    
    # 1. Δημιουργία συνθετικών δεδομένων
    print("\n📊 Δημιουργία συνθετικών δεδομένων...")
    data = generate_synthetic_data(days=14)  # 2 εβδομάδες δεδομένων
    print(f"   Δημιουργήθηκαν {len(data)} σημεία δεδομένων")
    print(f"   Εύρος γλυκόζης: {data['cgm'].min():.1f} - {data['cgm'].max():.1f} mg/dL")
    
    # 2. Προετοιμασία features
    print("\n🔧 Προετοιμασία features...")
    features_data = prepare_features(data)
    print(f"   Δημιουργήθηκαν {features_data.shape[1]} features")
    
    # 3. Διαχωρισμός σε train/test
    split_idx = int(len(features_data) * 0.8)
    train_data = features_data.iloc[:split_idx]
    test_data = features_data.iloc[split_idx:]
    
    # Προετοιμασία X και y
    feature_columns = [col for col in features_data.columns if col != 'cgm']
    X_train, y_train = train_data[feature_columns], train_data['cgm']
    X_test, y_test = test_data[feature_columns], test_data['cgm']
    
    print(f"   Train: {len(train_data)} samples")
    print(f"   Test: {len(test_data)} samples")
    
    # 4. Εκπαίδευση μοντέλων
    print("\n🧠 Εκπαίδευση μοντέλων...")
    
    models = {}
    results = {}
    
    # ARIMA (baseline)
    print("   Εκπαίδευση ARIMA...")
    try:
        arima_model = ARIMAModel(auto_arima=False, order=(2, 1, 1))
        arima_model.fit(X_train, y_train)
        arima_pred = arima_model.predict(X_test, horizon=30)
        models['ARIMA'] = arima_model
        results['ARIMA'] = arima_pred
        print("   ✅ ARIMA εκπαιδεύτηκε")
    except Exception as e:
        print(f"   ❌ ARIMA σφάλμα: {e}")
    
    # LSTM
    print("   Εκπαίδευση LSTM...")
    try:
        lstm_model = LSTMModel(sequence_length=12, hidden_size=32, epochs=20, patience=5)
        lstm_model.fit(X_train, y_train)
        lstm_pred = lstm_model.predict(X_test, horizon=30)
        models['LSTM'] = lstm_model
        results['LSTM'] = lstm_pred
        print("   ✅ LSTM εκπαιδεύτηκε")
    except Exception as e:
        print(f"   ❌ LSTM σφάλμα: {e}")
    
    # Transformer
    print("   Εκπαίδευση Transformer...")
    try:
        transformer_model = TransformerModel(
            sequence_length=12, d_model=64, num_layers=2, 
            epochs=15, patience=5, hybrid=False
        )
        transformer_model.fit(X_train, y_train)
        transformer_pred = transformer_model.predict(X_test, horizon=30)
        models['Transformer'] = transformer_model
        results['Transformer'] = transformer_pred
        print("   ✅ Transformer εκπαιδεύτηκε")
    except Exception as e:
        print(f"   ❌ Transformer σφάλμα: {e}")
    
    # Mechanistic Model
    print("   Εκπαίδευση Mechanistic...")
    try:
        mechanistic_model = MechanisticModel(body_weight=70)
        mechanistic_model.fit(X_train, y_train)
        mechanistic_pred = mechanistic_model.predict(X_test, horizon=30, initial_glucose=y_test.iloc[0])
        models['Mechanistic'] = mechanistic_model
        results['Mechanistic'] = mechanistic_pred
        print("   ✅ Mechanistic εκπαιδεύτηκε")
    except Exception as e:
        print(f"   ❌ Mechanistic σφάλμα: {e}")
    
    # 5. Αξιολόγηση μοντέλων
    print("\n📈 Αξιολόγηση μοντέλων...")
    
    metrics_results = {}
    
    for model_name, predictions in results.items():
        try:
            # Προσαρμογή μεγέθους προβλέψεων
            min_length = min(len(y_test), len(predictions))
            y_test_adj = y_test.iloc[:min_length].values
            pred_adj = predictions[:min_length]
            
            # Υπολογισμός μετρικών
            metrics = calculate_metrics(
                y_test_adj, pred_adj, 
                metrics=['rmse', 'mae', 'mape', 'clarke_grid', 'time_in_range']
            )
            metrics_results[model_name] = metrics
            
            print(f"\n   📊 {model_name}:")
            print(f"      RMSE: {metrics['rmse']:.2f} mg/dL")
            print(f"      MAE:  {metrics['mae']:.2f} mg/dL")
            print(f"      MAPE: {metrics['mape']:.2f}%")
            print(f"      Clarke Zone A: {metrics['clarke_grid']['A']:.1f}%")
            print(f"      Time in Range: {metrics['time_in_range']['in_range_percent']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Σφάλμα αξιολόγησης {model_name}: {e}")
    
    # 6. Οπτικοποιήσεις
    print("\n🎨 Δημιουργία γραφημάτων...")
    
    try:
        # Γράφημα σύγκρισης προβλέψεων
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (model_name, predictions) in enumerate(results.items()):
            if i < 4:  # Μέχρι 4 μοντέλα
                min_length = min(len(y_test), len(predictions))
                y_test_plot = y_test.iloc[:min_length].values
                pred_plot = predictions[:min_length]
                
                axes[i].plot(y_test_plot, 'b-', label='Πραγματικές', alpha=0.7)
                axes[i].plot(pred_plot, 'r--', label='Προβλέψεις', alpha=0.7)
                axes[i].set_title(f'{model_name}')
                axes[i].set_ylabel('Γλυκόζη (mg/dL)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Ζώνες γλυκόζης
                axes[i].axhspan(70, 180, alpha=0.1, color='green')
        
        plt.tight_layout()
        plt.suptitle('Σύγκριση Προβλέψεων Μοντέλων', fontsize=16, y=1.02)
        plt.show()
        
        # Γράφημα σύγκρισης μετρικών
        if len(metrics_results) > 1:
            fig_metrics = plot_metrics_comparison(metrics_results, title="Σύγκριση Μετρικών")
            plt.show()
        
        print("   ✅ Γραφήματα δημιουργήθηκαν")
        
    except Exception as e:
        print(f"   ❌ Σφάλμα γραφημάτων: {e}")
    
    # 7. Παράδειγμα χρήσης DigitalTwin wrapper
    print("\n🎯 Παράδειγμα DigitalTwin wrapper...")
    
    try:
        # Δημιουργία high-level wrapper
        twin = DigitalTwin(model_type='lstm', model_params={'epochs': 10, 'hidden_size': 32})
        twin.fit(X_train, y_train)
        twin_predictions = twin.predict(X_test, horizon=60)
        
        # Αξιολόγηση
        twin_metrics = twin.evaluate(X_test, y_test)
        print(f"   DigitalTwin RMSE: {twin_metrics['rmse']:.2f} mg/dL")
        
        # Αποθήκευση μοντέλου
        twin.save_model('digital_twin_model.pkl')
        print("   ✅ Μοντέλο αποθηκεύτηκε")
        
        # Φόρτωση μοντέλου
        loaded_twin = DigitalTwin.load_model('digital_twin_model.pkl')
        print("   ✅ Μοντέλο φορτώθηκε")
        
    except Exception as e:
        print(f"   ❌ DigitalTwin σφάλμα: {e}")
    
    print("\n🎉 Παράδειγμα ολοκληρώθηκε!")
    print("\nΕπόμενα βήματα:")
    print("- Δοκιμάστε με πραγματικά δεδομένα CGM")
    print("- Ρυθμίστε τις παραμέτρους των μοντέλων")
    print("- Προσθέστε περισσότερα features")
    print("- Χρησιμοποιήστε τα εργαλεία οπτικοποίησης")


if __name__ == "__main__":
    main() 