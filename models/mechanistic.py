"""
Μηχανιστικό μοντέλο για πρόβλεψη γλυκόζης.

Βασίζεται σε φυσιολογικά μοντέλα όπως το UVA/Padova και Cambridge (Hovorka).
Περιγράφει μαθηματικά τον μεταβολισμό γλυκόζης-ινσουλίνης.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.twin import BaseModel


class UVAPadovaModel(BaseModel):
    """
    Απλουστευμένη υλοποίηση του UVA/Padova μοντέλου.
    
    Το μοντέλο περιγράφει:
    - Την απορρόφηση γλυκόζης από το στομάχι
    - Την κινητική της ινσουλίνης
    - Την παραγωγή και χρήση γλυκόζης
    
    Βασίζεται στις εργασίες των Dalla Man et al. και Cobelli et al.
    """
    
    def __init__(self,
                 patient_params: Optional[Dict] = None,
                 simulation_step: float = 1.0,  # λεπτά
                 body_weight: float = 70.0):     # kg
        """
        Αρχικοποίηση μηχανιστικού μοντέλου.
        
        Args:
            patient_params: Φυσιολογικές παράμετροι ασθενούς
            simulation_step: Βήμα προσομοίωσης σε λεπτά
            body_weight: Βάρος σώματος σε kg
        """
        self.simulation_step = simulation_step
        self.body_weight = body_weight
        self.is_fitted = False
        self.feature_names = None
        
        # Προεπιλεγμένες παράμετροι (μέσος ασθενής)
        self.default_params = {
            # Γλυκόζη-ινσουλίνη
            'Vg': 1.88,      # Όγκος διανομής γλυκόζης (dL/kg)
            'k1': 0.065,     # Σταθερά μεταφοράς γλυκόζης (1/min)
            'k2': 0.079,     # Σταθερά μεταφοράς γλυκόζης (1/min)
            'Vi': 0.05,      # Όγκος διανομής ινσουλίνης (L/kg)
            'ke': 0.0079,    # Σταθερά απομάκρυνσης ινσουλίνης (1/min)
            'kd': 0.0164,    # Σταθερά αποδόμησης ινσουλίνης (1/min)
            
            # Ευαισθησία στην ινσουλίνη
            'SI': 0.00013,   # Ευαισθησία ινσουλίνης ((dL/kg)/min per μU/mL)
            'SG': 0.0021,    # Αποτελεσματικότητα γλυκόζης (1/min)
            
            # Παραγωγή γλυκόζης
            'EGP0': 0.0161,  # Βασική παραγωγή γλυκόζης (mg/kg/min)
            'kp1': 2.70,     # Παράμετρος παραγωγής γλυκόζης
            'kp2': 0.0021,   # Παράμετρος παραγωγής γλυκόζης
            'kp3': 0.009,    # Παράμετρος παραγωγής γλυκόζης
            
            # Απορρόφηση γευμάτων
            'kabs': 0.012,   # Σταθερά απορρόφησης γευμάτων (1/min)
            'kgri': 0.0558,  # Σταθερά κένωσης στομάχου (1/min)
            'kempt': 0.0496, # Σταθερά κένωσης στομάχου (1/min)
            
            # Ινσουλίνη
            'ka1': 0.006,    # Σταθερά απορρόφησης βραδείας ινσουλίνης (1/min)
            'ka2': 0.06,     # Σταθερά απορρόφησης γρήγορης ινσουλίνης (1/min)
            'kd': 0.0164,    # Σταθερά αποδόμησης ινσουλίνης (1/min)
        }
        
        # Συνδυασμός με παραμέτρους χρήστη
        self.params = self.default_params.copy()
        if patient_params:
            self.params.update(patient_params)
        
        # Αρχικές συνθήκες
        self.initial_state = None
        self.glucose_target = 100.0  # mg/dL
        
    def _glucose_insulin_model(self, state: np.ndarray, t: float, 
                              meal_input: float, insulin_input: float) -> np.ndarray:
        """
        Σύστημα διαφορικών εξισώσεων για το μοντέλο γλυκόζης-ινσουλίνης.
        
        Args:
            state: Τρέχουσα κατάσταση [Gp, Gt, I, X, Qsto1, Qsto2, Qgut, Isc1, Isc2]
            t: Χρόνος
            meal_input: Εισροή υδατανθράκων (mg/min)
            insulin_input: Εισροή ινσουλίνης (mU/min)
            
        Returns:
            Παράγωγοι της κατάστασης
        """
        # Αποδόμηση κατάστασης
        Gp, Gt, I, X, Qsto1, Qsto2, Qgut, Isc1, Isc2 = state
        
        # Υπολογισμός γλυκόζης πλάσματος (mg/dL)
        G = Gp / (self.params['Vg'] * self.body_weight)
        
        # Παραγωγή γλυκόζης από το ήπαρ
        EGP = max(0, self.params['EGP0'] * (1 - self.params['kp1'] * X) 
                  - self.params['kp2'] * G - self.params['kp3'] * I)
        
        # Χρήση γλυκόζης
        Uii = max(0, self.params['SG'] * Gt)
        Uid = max(0, X * G)
        
        # Απορρόφηση γευμάτων
        Qsto = Qsto1 + Qsto2
        Kempt = self.params['kempt']
        if Qsto > 0:
            alpha = 5 / (2 * (1 - 0.01))
            beta = 5 / (2 * 0.01)
            Kempt = self.params['kempt'] * (alpha * (Qsto**beta)) / (alpha + (Qsto**beta))
        
        Ra = self.params['kabs'] * Qgut / self.body_weight
        
        # Δυναμική ινσουλίνης
        Isc = Isc1 + Isc2
        insulin_absorption = self.params['ka1'] * Isc1 + self.params['ka2'] * Isc2
        
        # Διαφορικές εξισώσεις
        dGp_dt = EGP + Ra - Uii - (self.params['k1'] * Gp) + (self.params['k2'] * Gt)
        dGt_dt = (self.params['k1'] * Gp) - (self.params['k2'] * Gt) - Uid
        dI_dt = insulin_absorption / (self.params['Vi'] * self.body_weight) - self.params['ke'] * I
        dX_dt = self.params['SI'] * I - self.params['SG'] * X
        
        # Γεύματα
        dQsto1_dt = meal_input - self.params['kgri'] * Qsto1
        dQsto2_dt = self.params['kgri'] * Qsto1 - Kempt * Qsto2
        dQgut_dt = Kempt * Qsto2 - self.params['kabs'] * Qgut
        
        # Ινσουλίνη
        dIsc1_dt = insulin_input * 0.7 - self.params['ka1'] * Isc1
        dIsc2_dt = insulin_input * 0.3 - self.params['ka2'] * Isc2
        
        return np.array([dGp_dt, dGt_dt, dI_dt, dX_dt, dQsto1_dt, dQsto2_dt, dQgut_dt, dIsc1_dt, dIsc2_dt])
    
    def _get_steady_state(self, glucose_level: float = 100.0) -> np.ndarray:
        """Υπολογισμός steady-state για δεδομένο επίπεδο γλυκόζης."""
        # Steady state values
        G = glucose_level  # mg/dL
        Vg = self.params['Vg'] * self.body_weight
        Gp = G * Vg
        Gt = Gp
        
        # Ισορροπία ινσουλίνης
        EGP_ss = self.params['EGP0'] * self.body_weight
        Uii_ss = self.params['SG'] * Gt
        I_ss = max(0, (EGP_ss - Uii_ss) / (self.params['SI'] * G * self.body_weight))
        X_ss = self.params['SI'] * I_ss / self.params['SG']
        
        # Κενά γεύματα και ινσουλίνη
        Qsto1_ss = 0.0
        Qsto2_ss = 0.0
        Qgut_ss = 0.0
        Isc1_ss = 0.0
        Isc2_ss = 0.0
        
        return np.array([Gp, Gt, I_ss, X_ss, Qsto1_ss, Qsto2_ss, Qgut_ss, Isc1_ss, Isc2_ss])
    
    def simulate(self, 
                 duration: int,
                 meals: Optional[List[Tuple[float, float]]] = None,
                 insulin: Optional[List[Tuple[float, float]]] = None,
                 initial_glucose: float = 100.0) -> pd.DataFrame:
        """
        Προσομοίωση του συστήματος γλυκόζης-ινσουλίνης.
        
        Args:
            duration: Διάρκεια προσομοίωσης σε λεπτά
            meals: Λίστα γευμάτων [(χρόνος, υδατάνθρακες_σε_g)]
            insulin: Λίστα δόσεων ινσουλίνης [(χρόνος, μονάδες)]
            initial_glucose: Αρχική γλυκόζη σε mg/dL
            
        Returns:
            DataFrame με τα αποτελέσματα της προσομοίωσης
        """
        # Χρονικά βήματα
        time_points = np.arange(0, duration + self.simulation_step, self.simulation_step)
        
        # Αρχικές συνθήκες
        initial_state = self._get_steady_state(initial_glucose)
        
        # Προετοιμασία εισροών
        meal_inputs = np.zeros(len(time_points))
        insulin_inputs = np.zeros(len(time_points))
        
        if meals:
            for meal_time, carbs in meals:
                if 0 <= meal_time < duration:
                    idx = int(meal_time / self.simulation_step)
                    if idx < len(meal_inputs):
                        # Μετατροπή g υδατανθράκων σε mg/min για ~15 λεπτά
                        meal_inputs[idx:idx+15] += (carbs * 1000) / 15
        
        if insulin:
            for insulin_time, units in insulin:
                if 0 <= insulin_time < duration:
                    idx = int(insulin_time / self.simulation_step)
                    if idx < len(insulin_inputs):
                        # Μετατροπή μονάδων σε mU/min για ~5 λεπτά (bolus)
                        insulin_inputs[idx:idx+5] += (units * 1000) / 5
        
        # Προσομοίωση
        results = []
        current_state = initial_state.copy()
        
        for i, t in enumerate(time_points):
            # Ενημέρωση εισροών
            meal_input = meal_inputs[i] if i < len(meal_inputs) else 0
            insulin_input = insulin_inputs[i] if i < len(insulin_inputs) else 0
            
            # Ενοποίηση για ένα βήμα
            if i < len(time_points) - 1:
                t_span = [t, t + self.simulation_step]
                sol = odeint(self._glucose_insulin_model, current_state, t_span, 
                           args=(meal_input, insulin_input))
                current_state = sol[-1]
            
            # Υπολογισμός γλυκόζης πλάσματος
            glucose = current_state[0] / (self.params['Vg'] * self.body_weight)
            insulin_plasma = current_state[2]
            
            results.append({
                'time': t,
                'glucose': glucose,
                'insulin': insulin_plasma,
                'meal_input': meal_input,
                'insulin_input': insulin_input
            })
        
        return pd.DataFrame(results)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'UVAPadovaModel':
        """
        Προσαρμογή παραμέτρων του μοντέλου στα δεδομένα.
        
        Args:
            X: Features που περιλαμβάνουν γεύματα, ινσουλίνη κλπ
            y: CGM μετρήσεις
            **kwargs: Επιπλέον παράμετροι
        """
        self.feature_names = X.columns.tolist()
        
        # Εδώ θα μπορούσαμε να υλοποιήσουμε parameter estimation
        # Για απλότητα, χρησιμοποιούμε προεπιλεγμένες παραμέτρους
        
        # Υπολογισμός αρχικής κατάστασης από τα δεδομένα
        initial_glucose = y.iloc[0] if len(y) > 0 else 100.0
        self.initial_state = self._get_steady_state(initial_glucose)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, horizon: int = 30, **kwargs) -> np.ndarray:
        """
        Πρόβλεψη γλυκόζης χρησιμοποιώντας το μηχανιστικό μοντέλο.
        
        Args:
            X: Features με πληροφορίες γευμάτων/ινσουλίνης
            horizon: Ορίζοντας πρόβλεψης σε λεπτά
            **kwargs: Επιπλέον παράμετροι
            
        Returns:
            Προβλέψεις γλυκόζης
        """
        if not self.is_fitted:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί")
        
        # Εξαγωγή πληροφοριών γευμάτων και ινσουλίνης από X
        meals = []
        insulin = []
        
        # Υποθέτουμε ότι το X περιέχει στήλες για γεύματα και ινσουλίνη
        if 'carbs' in X.columns:
            for i, row in X.iterrows():
                if row['carbs'] > 0:
                    meals.append((i * 5, row['carbs']))  # Υποθέτουμε 5-min intervals
        
        if 'insulin' in X.columns:
            for i, row in X.iterrows():
                if row['insulin'] > 0:
                    insulin.append((i * 5, row['insulin']))
        
        # Αρχική γλυκόζη (τελευταία γνωστή τιμή)
        initial_glucose = kwargs.get('initial_glucose', 100.0)
        
        # Προσομοίωση
        simulation_results = self.simulate(
            duration=horizon,
            meals=meals,
            insulin=insulin,
            initial_glucose=initial_glucose
        )
        
        # Επιστροφή προβλέψεων σε intervals των 5 λεπτών
        prediction_times = np.arange(5, horizon + 1, 5)
        predictions = []
        
        for t in prediction_times:
            # Παρεμβολή για να βρούμε την τιμή σε συγκεκριμένο χρόνο
            idx = np.searchsorted(simulation_results['time'], t)
            if idx < len(simulation_results):
                predictions.append(simulation_results.iloc[idx]['glucose'])
            else:
                predictions.append(simulation_results.iloc[-1]['glucose'])
        
        return np.array(predictions)
    
    def get_params(self) -> Dict[str, Any]:
        """Επιστροφή παραμέτρων του μοντέλου."""
        return {
            'patient_params': self.params.copy(),
            'simulation_step': self.simulation_step,
            'body_weight': self.body_weight,
            'glucose_target': self.glucose_target
        }
    
    def set_params(self, **params) -> 'UVAPadovaModel':
        """Ορισμός παραμέτρων του μοντέλου."""
        if 'patient_params' in params:
            self.params.update(params['patient_params'])
        
        for key, value in params.items():
            if hasattr(self, key) and key != 'patient_params':
                setattr(self, key, value)
        
        return self
    
    def get_simulation_details(self) -> Dict[str, Any]:
        """Επιστροφή λεπτομερών πληροφοριών προσομοίωσης."""
        return {
            'model_type': 'UVA/Padova',
            'parameters': self.params,
            'body_weight': self.body_weight,
            'simulation_step': self.simulation_step,
            'initial_state': self.initial_state.tolist() if self.initial_state is not None else None
        }


# Alias για συμβατότητα
MechanisticModel = UVAPadovaModel 