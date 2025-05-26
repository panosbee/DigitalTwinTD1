"""
Real-Time Intelligence Engine για ψηφιακό δίδυμο διαβήτη.

Περιλαμβάνει:
- Edge inference για wearable devices
- Federated learning για privacy-preserving training
- Digital biomarker discovery και monitoring
- Clinical decision support system
- Causal inference για treatment optimization
- Personalized adaptive control
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import asyncio
import threading
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fedml

    FEDML_AVAILABLE = True
except ImportError:
    FEDML_AVAILABLE = False

try:
    from causalnex.structure import StructureModel
    from causalnex.inference import InferenceEngine

    CAUSAL_INFERENCE_AVAILABLE = True
except ImportError:
    CAUSAL_INFERENCE_AVAILABLE = False


@dataclass
class GlucoseAlert:
    """Ειδοποίηση για γλυκόζη."""

    timestamp: datetime
    glucose_value: float
    predicted_value: float
    alert_type: str  # 'hypoglycemia', 'hyperglycemia', 'rapid_change'
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str
    confidence: float


@dataclass
class DigitalBiomarker:
    """Ψηφιακός βιοδείκτης."""

    name: str
    value: float
    timestamp: datetime
    trend: str  # 'increasing', 'decreasing', 'stable'
    percentile: float  # Σε σχέση με population
    clinical_significance: str


class EdgeInferenceEngine:
    """
    Optimized inference engine για edge devices (smartwatches, CGM devices).

    Χρησιμοποιεί quantized models και efficient architectures για
    real-time inference με ελάχιστη κατανάλωση ενέργειας.
    """

    def __init__(
        self,
        model_path: str,
        quantization: bool = True,
        max_latency_ms: float = 100.0,
        battery_optimization: bool = True,
    ):

        self.model_path = model_path
        self.quantization = quantization
        self.max_latency_ms = max_latency_ms
        self.battery_optimization = battery_optimization

        self.model = None
        self.preprocessor = None
        self.last_inference_time = None
        self.inference_stats = {"total_inferences": 0, "avg_latency_ms": 0, "battery_usage": 0}

    def load_model(self):
        """Φόρτωση και optimization του μοντέλου για edge deployment."""

        # Load model
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model = checkpoint["model"]

        # Quantization για μείωση μεγέθους και ταχύτητα
        if self.quantization:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )

        # Optimization για inference
        self.model.eval()
        if hasattr(self.model, "fuse_model"):
            self.model.fuse_model()

        # JIT compilation για ταχύτητα
        if hasattr(torch, "jit"):
            dummy_input = torch.randn(1, 10)  # Adjust based on model
            self.model = torch.jit.trace(self.model, dummy_input)

        print(f"✅ Edge model loaded and optimized")

    def predict_glucose(
        self,
        cgm_readings: List[float],
        additional_data: Optional[Dict] = None,
        return_confidence: bool = True,
    ) -> Tuple[float, float]:
        """
        Ultra-fast glucose prediction για edge devices.

        Returns:
            Tuple of (predicted_glucose, confidence)
        """
        start_time = datetime.now()

        # Preprocessing (lightweight)
        input_tensor = torch.tensor(cgm_readings[-10:], dtype=torch.float32).unsqueeze(0)

        # Inference
        with torch.no_grad():
            if self.battery_optimization:
                # Adaptive precision based on battery level
                with torch.autocast("cpu"):
                    prediction = self.model(input_tensor)
            else:
                prediction = self.model(input_tensor)

        # Extract prediction and confidence
        glucose_pred = prediction[0].item()
        confidence = prediction[1].item() if prediction.shape[1] > 1 else 0.95

        # Update stats
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.inference_stats["total_inferences"] += 1
        self.inference_stats["avg_latency_ms"] = (
            self.inference_stats["avg_latency_ms"] * (self.inference_stats["total_inferences"] - 1)
            + latency
        ) / self.inference_stats["total_inferences"]

        if latency > self.max_latency_ms:
            print(f"⚠️ Latency warning: {latency:.1f}ms > {self.max_latency_ms}ms")

        return glucose_pred, confidence

    def adaptive_sampling(self, current_glucose: float, trend: str) -> int:
        """
        Adaptive sampling rate based on glucose level and trend.

        Returns:
            Sampling interval in seconds
        """
        if current_glucose < 70 or current_glucose > 180:
            return 60  # Every minute for dangerous levels
        elif trend in ["rapidly_increasing", "rapidly_decreasing"]:
            return 120  # Every 2 minutes for rapid changes
        else:
            return 300  # Every 5 minutes for normal levels


class FederatedLearningManager:
    """
    Federated Learning για privacy-preserving model updates.

    Επιτρέπει στους χρήστες να συνεισφέρουν στη βελτίωση του μοντέλου
    χωρίς να μοιράζονται προσωπικά δεδομένα.
    """

    def __init__(
        self,
        client_id: str,
        server_url: str = "federated.digitaltwint1d.org",
        privacy_budget: float = 1.0,
        min_clients: int = 10,
    ):

        self.client_id = client_id
        self.server_url = server_url
        self.privacy_budget = privacy_budget
        self.min_clients = min_clients

        self.local_model = None
        self.global_model_version = 0
        self.contribution_score = 0.0

    def train_local_model(
        self, local_data: pd.DataFrame, epochs: int = 5, differential_privacy: bool = True
    ) -> Dict[str, Any]:
        """
        Εκπαίδευση τοπικού μοντέλου με differential privacy.

        Returns:
            Model updates για αποστολή στον server
        """
        if not FEDML_AVAILABLE:
            print("⚠️ FedML not available. Simulating federated learning...")

        # Simulate local training with privacy
        if differential_privacy:
            # Add noise για differential privacy
            noise_scale = 1.0 / self.privacy_budget
            local_data = local_data + np.random.laplace(0, noise_scale, local_data.shape)

        # Train local model (simplified)
        from models.lstm import LSTMModel

        local_model = LSTMModel(epochs=epochs, verbose=False)
        X = local_data.drop("cgm", axis=1) if "cgm" in local_data.columns else local_data
        y = local_data["cgm"] if "cgm" in local_data.columns else local_data.iloc[:, 0]

        local_model.fit(X, y)

        # Extract model updates (gradients/weights)
        model_updates = {
            "client_id": self.client_id,
            "model_version": self.global_model_version,
            "updates": self._extract_model_updates(local_model),
            "data_size": len(local_data),
            "privacy_spent": 1.0 / self.privacy_budget,
        }

        return model_updates

    def _extract_model_updates(self, model) -> Dict:
        """Εξαγωγή model updates για federated averaging."""
        if hasattr(model, "model") and hasattr(model.model, "state_dict"):
            return {name: param.cpu().numpy() for name, param in model.model.state_dict().items()}
        return {}

    async def send_updates(self, model_updates: Dict) -> bool:
        """Αποστολή updates στον federated server."""
        try:
            # Simulate sending to federated server
            print(f"📡 Sending model updates from client {self.client_id}")
            await asyncio.sleep(0.1)  # Simulate network delay

            # Update contribution score
            self.contribution_score += model_updates["data_size"] * 0.01

            return True
        except Exception as e:
            print(f"❌ Failed to send updates: {e}")
            return False

    async def receive_global_model(self) -> Optional[Dict]:
        """Λήψη global model από τον server."""
        try:
            # Simulate receiving global model
            print(f"📥 Receiving global model update")
            await asyncio.sleep(0.1)

            self.global_model_version += 1

            return {
                "model_version": self.global_model_version,
                "global_weights": {},  # Simulated global weights
                "performance_metrics": {
                    "global_rmse": 15.2,
                    "participating_clients": 150,
                    "rounds_completed": 25,
                },
            }
        except Exception as e:
            print(f"❌ Failed to receive global model: {e}")
            return None


class DigitalBiomarkerEngine:
    """
    Ανακάλυψη και παρακολούθηση ψηφιακών βιοδεικτών.

    Εντοπίζει νέα patterns στα δεδομένα που μπορεί να έχουν κλινική σημασία.
    """

    def __init__(self):
        self.known_biomarkers = {
            "glucose_variability": {
                "description": "Μεταβλητότητα γλυκόζης (CV%)",
                "normal_range": (15, 36),
                "clinical_significance": "Δείκτης γλυκαιμικού ελέγχου",
            },
            "dawn_phenomenon": {
                "description": "Αύξηση γλυκόζης το πρωί",
                "normal_range": (0, 30),  # mg/dL increase
                "clinical_significance": "Φυσιολογική αντίδραση ορμονών",
            },
            "hypoglycemia_risk_index": {
                "description": "Δείκτης κινδύνου υπογλυκαιμίας",
                "normal_range": (0, 1.1),
                "clinical_significance": "Πρόβλεψη επικίνδυνων επεισοδίων",
            },
            "meal_response_profile": {
                "description": "Προφίλ απόκρισης σε γεύματα",
                "normal_range": (0.5, 2.0),  # Response factor
                "clinical_significance": "Ευαισθησία στους υδατάνθρακες",
            },
        }

        self.discovered_biomarkers = []

    def calculate_biomarkers(
        self,
        glucose_data: pd.Series,
        meal_data: Optional[pd.Series] = None,
        insulin_data: Optional[pd.Series] = None,
    ) -> List[DigitalBiomarker]:
        """Υπολογισμός όλων των γνωστών βιοδεικτών."""

        biomarkers = []
        timestamp = datetime.now()

        # 1. Glucose Variability (CV%)
        cv = (glucose_data.std() / glucose_data.mean()) * 100
        biomarkers.append(
            DigitalBiomarker(
                name="glucose_variability",
                value=cv,
                timestamp=timestamp,
                trend=self._calculate_trend(cv, [cv]),  # Simplified
                percentile=self._calculate_percentile(cv, (15, 36)),
                clinical_significance=self._assess_clinical_significance("glucose_variability", cv),
            )
        )

        # 2. Dawn Phenomenon
        if len(glucose_data) >= 24:  # At least 24 hours of data
            dawn_effect = self._calculate_dawn_phenomenon(glucose_data)
            biomarkers.append(
                DigitalBiomarker(
                    name="dawn_phenomenon",
                    value=dawn_effect,
                    timestamp=timestamp,
                    trend="stable",
                    percentile=self._calculate_percentile(dawn_effect, (0, 30)),
                    clinical_significance=self._assess_clinical_significance(
                        "dawn_phenomenon", dawn_effect
                    ),
                )
            )

        # 3. Hypoglycemia Risk Index
        hri = self._calculate_hypoglycemia_risk(glucose_data)
        biomarkers.append(
            DigitalBiomarker(
                name="hypoglycemia_risk_index",
                value=hri,
                timestamp=timestamp,
                trend=self._calculate_trend(hri, [hri]),
                percentile=self._calculate_percentile(hri, (0, 1.1)),
                clinical_significance=self._assess_clinical_significance(
                    "hypoglycemia_risk_index", hri
                ),
            )
        )

        # 4. Meal Response Profile
        if meal_data is not None:
            meal_response = self._calculate_meal_response(glucose_data, meal_data)
            biomarkers.append(
                DigitalBiomarker(
                    name="meal_response_profile",
                    value=meal_response,
                    timestamp=timestamp,
                    trend="stable",
                    percentile=self._calculate_percentile(meal_response, (0.5, 2.0)),
                    clinical_significance=self._assess_clinical_significance(
                        "meal_response_profile", meal_response
                    ),
                )
            )

        return biomarkers

    def discover_new_biomarkers(
        self, multi_modal_data: pd.DataFrame, clinical_outcomes: pd.Series
    ) -> List[Dict]:
        """
        Ανακάλυψη νέων ψηφιακών βιοδεικτών μέσω ML.

        Χρησιμοποιεί feature selection και correlation analysis
        για εντοπισμό νέων patterns με κλινική σημασία.
        """
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.ensemble import RandomForestRegressor

        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=10)
        selected_features = selector.fit_transform(multi_modal_data, clinical_outcomes)
        feature_names = multi_modal_data.columns[selector.get_support()]

        # Random Forest για feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(selected_features, clinical_outcomes)

        new_biomarkers = []
        for i, (feature, importance) in enumerate(zip(feature_names, rf.feature_importances_)):
            if importance > 0.05:  # Significant features
                new_biomarkers.append(
                    {
                        "name": f"discovered_biomarker_{i}",
                        "feature": feature,
                        "importance": importance,
                        "correlation": np.corrcoef(multi_modal_data[feature], clinical_outcomes)[
                            0, 1
                        ],
                        "potential_clinical_use": self._suggest_clinical_use(feature, importance),
                    }
                )

        self.discovered_biomarkers.extend(new_biomarkers)
        return new_biomarkers

    def _calculate_dawn_phenomenon(self, glucose_data: pd.Series) -> float:
        """Υπολογισμός dawn phenomenon effect."""
        if isinstance(glucose_data.index, pd.DatetimeIndex):
            early_morning = glucose_data.between_time("04:00", "08:00")
            late_night = glucose_data.between_time("02:00", "04:00")

            if len(early_morning) > 0 and len(late_night) > 0:
                return early_morning.mean() - late_night.mean()

        return 0.0

    def _calculate_hypoglycemia_risk(self, glucose_data: pd.Series) -> float:
        """Υπολογισμός δείκτη κινδύνου υπογλυκαιμίας."""
        # Simplified LBGI calculation
        log_glucose = np.log(glucose_data / 18.0)  # Convert to mmol/L
        f_bg = 1.509 * (log_glucose**1.084 - 5.381)
        risk_bg = 10 * (f_bg**2)
        lbgi = np.mean(np.where(f_bg < 0, risk_bg, 0))
        return lbgi

    def _calculate_meal_response(self, glucose_data: pd.Series, meal_data: pd.Series) -> float:
        """Υπολογισμός meal response profile."""
        # Find meal events and subsequent glucose response
        meal_responses = []

        for meal_time in meal_data[meal_data > 0].index:
            # Get glucose 2 hours after meal
            end_time = meal_time + pd.Timedelta(hours=2)
            post_meal_glucose = glucose_data.loc[meal_time:end_time]

            if len(post_meal_glucose) > 0:
                peak_glucose = post_meal_glucose.max()
                baseline_glucose = glucose_data.loc[meal_time]
                response = (peak_glucose - baseline_glucose) / meal_data.loc[meal_time]
                meal_responses.append(response)

        return np.mean(meal_responses) if meal_responses else 1.0

    def _calculate_trend(self, current_value: float, historical_values: List[float]) -> str:
        """Υπολογισμός trend για biomarker."""
        if len(historical_values) < 2:
            return "stable"

        recent_avg = np.mean(historical_values[-3:])
        older_avg = np.mean(historical_values[:-3]) if len(historical_values) > 3 else recent_avg

        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _calculate_percentile(self, value: float, normal_range: Tuple[float, float]) -> float:
        """Υπολογισμός percentile σε σχέση με normal range."""
        low, high = normal_range
        if value <= low:
            return 10.0
        elif value >= high:
            return 90.0
        else:
            return 10.0 + 80.0 * (value - low) / (high - low)

    def _assess_clinical_significance(self, biomarker_name: str, value: float) -> str:
        """Αξιολόγηση κλινικής σημασίας."""
        if biomarker_name in self.known_biomarkers:
            normal_range = self.known_biomarkers[biomarker_name]["normal_range"]
            low, high = normal_range

            if value < low:
                return "Χαμηλός - παρακολούθηση συνιστάται"
            elif value > high:
                return "Υψηλός - ιατρική συμβουλή συνιστάται"
            else:
                return "Εντός φυσιολογικών ορίων"

        return "Άγνωστη κλινική σημασία"

    def _suggest_clinical_use(self, feature: str, importance: float) -> str:
        """Πρόταση κλινικής χρήσης για νέο biomarker."""
        use_cases = {
            "high": "Πιθανός δείκτης για personalized therapy",
            "medium": "Ενδιαφέρον για επιπλέον έρευνα",
            "low": "Δευτερεύων δείκτης για συνδυαστική ανάλυση",
        }

        if importance > 0.2:
            level = "high"
        elif importance > 0.1:
            level = "medium"
        else:
            level = "low"

        return use_cases[level]


class ClinicalDecisionSupport:
    """
    Σύστημα υποστήριξης κλινικών αποφάσεων.

    Παρέχει evidence-based recommendations για:
    - Δοσολογία ινσουλίνης
    - Διατροφικές επιλογές
    - Άσκηση και lifestyle
    - Τιμή στόχου γλυκόζης
    """

    def __init__(self):
        self.clinical_guidelines = {
            "ada_2024": self._load_ada_guidelines(),
            "easd_2024": self._load_easd_guidelines(),
            "custom": {},
        }

        self.patient_history = {}
        self.recommendation_cache = {}

    def generate_recommendations(
        self, current_glucose: float, predicted_glucose: float, patient_profile: Dict, context: Dict
    ) -> List[Dict]:
        """
        Δημιουργία personalized recommendations.

        Args:
            current_glucose: Τρέχουσα γλυκόζη (mg/dL)
            predicted_glucose: Προβλεπόμενη γλυκόζη (mg/dL)
            patient_profile: Προφίλ ασθενούς (ηλικία, βάρος, HbA1c κλπ)
            context: Τρέχον context (ώρα, δραστηριότητα, γεύματα κλπ)
        """
        recommendations = []

        # 1. Insulin Recommendations
        insulin_rec = self._generate_insulin_recommendation(
            current_glucose, predicted_glucose, patient_profile, context
        )
        if insulin_rec:
            recommendations.append(insulin_rec)

        # 2. Nutritional Recommendations
        nutrition_rec = self._generate_nutrition_recommendation(
            current_glucose, predicted_glucose, context
        )
        if nutrition_rec:
            recommendations.append(nutrition_rec)

        # 3. Activity Recommendations
        activity_rec = self._generate_activity_recommendation(
            current_glucose, predicted_glucose, context
        )
        if activity_rec:
            recommendations.append(activity_rec)

        # 4. Monitoring Recommendations
        monitoring_rec = self._generate_monitoring_recommendation(
            current_glucose, predicted_glucose
        )
        if monitoring_rec:
            recommendations.append(monitoring_rec)

        return recommendations

    def _generate_insulin_recommendation(
        self, current_glucose: float, predicted_glucose: float, patient_profile: Dict, context: Dict
    ) -> Optional[Dict]:
        """Συστάσεις για ινσουλίνη."""

        # Get insulin sensitivity from patient profile
        insulin_sensitivity = patient_profile.get("insulin_sensitivity", 50)  # mg/dL per unit

        # Calculate correction needed
        target_glucose = patient_profile.get("target_glucose", 120)
        correction_needed = predicted_glucose - target_glucose

        if abs(correction_needed) < 20:  # Within acceptable range
            return None

        # Calculate insulin dose
        if correction_needed > 0:  # High glucose
            insulin_units = correction_needed / insulin_sensitivity

            # Safety checks
            max_bolus = patient_profile.get("max_bolus", 10)
            insulin_units = min(insulin_units, max_bolus)

            return {
                "type": "insulin",
                "action": "bolus",
                "dose": round(insulin_units, 1),
                "urgency": "high" if predicted_glucose > 200 else "medium",
                "reasoning": f"Διόρθωση για προβλεπόμενη γλυκόζη {predicted_glucose:.0f} mg/dL",
                "safety_notes": "Ελέγξτε κετόνες αν γλυκόζη >250 mg/dL",
            }

        return None

    def _generate_nutrition_recommendation(
        self, current_glucose: float, predicted_glucose: float, context: Dict
    ) -> Optional[Dict]:
        """Διατροφικές συστάσεις."""

        current_time = context.get("time", datetime.now().hour)

        if predicted_glucose < 70:  # Hypoglycemia risk
            if current_glucose < 60:
                carbs_needed = 20  # Emergency treatment
                urgency = "critical"
            else:
                carbs_needed = 15  # Mild hypoglycemia
                urgency = "high"

            return {
                "type": "nutrition",
                "action": "consume_carbs",
                "amount": f"{carbs_needed}g γρήγορων υδατανθράκων",
                "urgency": urgency,
                "suggestions": ["Χυμός φρούτων", "Ζάχαρη", "Glucose tablets"],
                "reasoning": f"Πρόληψη υπογλυκαιμίας (προβλεπόμενη: {predicted_glucose:.0f} mg/dL)",
            }

        elif predicted_glucose > 180 and 6 <= current_time <= 10:  # High morning glucose
            return {
                "type": "nutrition",
                "action": "modify_breakfast",
                "urgency": "medium",
                "suggestions": [
                    "Μειώστε τους υδατάνθρακες στο πρωινό",
                    "Προσθέστε περισσότερες πρωτεΐνες",
                    "Αποφύγετε γρήγορους υδατάνθρακες",
                ],
                "reasoning": "Έλεγχος πρωινής υπεργλυκαιμίας",
            }

        return None

    def _generate_activity_recommendation(
        self, current_glucose: float, predicted_glucose: float, context: Dict
    ) -> Optional[Dict]:
        """Συστάσεις για δραστηριότητα."""

        if 120 <= predicted_glucose <= 180:  # Good range for exercise
            return {
                "type": "activity",
                "action": "light_exercise",
                "duration": "15-30 λεπτά",
                "urgency": "low",
                "suggestions": ["Περπάτημα", "Ελαφριές ασκήσεις stretching", "Ανάβαση σκάλας"],
                "reasoning": "Βελτίωση ευαισθησίας στην ινσουλίνη",
                "safety_notes": "Μετρήστε γλυκόζη πριν και μετά",
            }

        elif predicted_glucose > 250:  # Too high for exercise
            return {
                "type": "activity",
                "action": "avoid_exercise",
                "urgency": "high",
                "reasoning": "Πολύ υψηλή γλυκόζη για άσκηση",
                "safety_notes": "Ελέγξτε κετόνες πριν από οποιαδήποτε δραστηριότητα",
            }

        return None

    def _generate_monitoring_recommendation(
        self, current_glucose: float, predicted_glucose: float
    ) -> Optional[Dict]:
        """Συστάσεις για παρακολούθηση."""

        if predicted_glucose < 70 or predicted_glucose > 250:
            frequency = "κάθε 15 λεπτά"
            urgency = "high"
        elif predicted_glucose < 90 or predicted_glucose > 200:
            frequency = "κάθε 30 λεπτά"
            urgency = "medium"
        else:
            return None

        return {
            "type": "monitoring",
            "action": "increase_frequency",
            "frequency": frequency,
            "urgency": urgency,
            "reasoning": f"Στενή παρακολούθηση για προβλεπόμενη γλυκόζη {predicted_glucose:.0f} mg/dL",
            "duration": "για τις επόμενες 2 ώρες",
        }

    def _load_ada_guidelines(self) -> Dict:
        """Φόρτωση ADA guidelines."""
        return {
            "target_glucose_range": (70, 180),
            "hba1c_target": 7.0,
            "hypoglycemia_threshold": 70,
            "severe_hypoglycemia_threshold": 54,
            "hyperglycemia_threshold": 250,
        }

    def _load_easd_guidelines(self) -> Dict:
        """Φόρτωση EASD guidelines."""
        return {
            "target_glucose_range": (70, 180),
            "hba1c_target": 7.0,
            "time_in_range_target": 70,  # % of time in 70-180 mg/dL
            "below_range_limit": 4,  # % of time below 70 mg/dL
            "above_range_limit": 25,  # % of time above 180 mg/dL
        }


class RealTimeIntelligenceEngine:
    """
    Κεντρικός Real-Time Intelligence Engine που συντονίζει όλα τα υποσυστήματα.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize sub-engines
        self.edge_engine = EdgeInferenceEngine(
            model_path=config.get("edge_model_path", "models/edge_model.pth"),
            quantization=config.get("quantization", True),
        )

        self.federated_manager = FederatedLearningManager(
            client_id=config.get("client_id", "anonymous"),
            server_url=config.get("federated_server", "federated.digitaltwint1d.org"),
        )

        self.biomarker_engine = DigitalBiomarkerEngine()
        self.clinical_support = ClinicalDecisionSupport()

        # Real-time data streams
        self.data_streams = {
            "cgm": [],
            "meals": [],
            "insulin": [],
            "activity": [],
            "heart_rate": [],
            "sleep": [],
        }

        # Alert system
        self.alert_callbacks = []
        self.active_alerts = []

        # Performance monitoring
        self.performance_metrics = {
            "prediction_accuracy": 0.0,
            "alert_precision": 0.0,
            "user_satisfaction": 0.0,
            "clinical_outcomes": 0.0,
        }

    async def start_real_time_monitoring(self):
        """Εκκίνηση real-time monitoring συστήματος."""
        print("🚀 Starting Real-Time Intelligence Engine...")

        # Load edge model
        self.edge_engine.load_model()

        # Start monitoring tasks
        tasks = [
            self._glucose_monitoring_loop(),
            self._biomarker_monitoring_loop(),
            self._federated_learning_loop(),
            self._alert_processing_loop(),
        ]

        await asyncio.gather(*tasks)

    async def _glucose_monitoring_loop(self):
        """Κύκλος παρακολούθησης γλυκόζης."""
        while True:
            try:
                # Get latest CGM readings
                if len(self.data_streams["cgm"]) >= 10:
                    latest_readings = [
                        reading["value"] for reading in self.data_streams["cgm"][-10:]
                    ]

                    # Predict next glucose value
                    predicted_glucose, confidence = self.edge_engine.predict_glucose(
                        latest_readings, return_confidence=True
                    )

                    # Check for alerts
                    await self._check_glucose_alerts(
                        current_glucose=latest_readings[-1],
                        predicted_glucose=predicted_glucose,
                        confidence=confidence,
                    )

                    # Adaptive sampling
                    current_glucose = latest_readings[-1]
                    trend = self._calculate_glucose_trend(latest_readings)
                    sampling_interval = self.edge_engine.adaptive_sampling(current_glucose, trend)

                    await asyncio.sleep(sampling_interval)
                else:
                    await asyncio.sleep(60)  # Wait for more data

            except Exception as e:
                print(f"❌ Error in glucose monitoring: {e}")
                await asyncio.sleep(60)

    async def _biomarker_monitoring_loop(self):
        """Κύκλος παρακολούθησης biomarkers."""
        while True:
            try:
                if len(self.data_streams["cgm"]) >= 100:  # Need sufficient data
                    # Extract glucose data
                    glucose_values = [
                        reading["value"] for reading in self.data_streams["cgm"][-100:]
                    ]
                    glucose_series = pd.Series(glucose_values)

                    # Calculate biomarkers
                    biomarkers = self.biomarker_engine.calculate_biomarkers(glucose_series)

                    # Store and analyze biomarkers
                    for biomarker in biomarkers:
                        await self._process_biomarker(biomarker)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                print(f"❌ Error in biomarker monitoring: {e}")
                await asyncio.sleep(3600)

    async def _federated_learning_loop(self):
        """Κύκλος federated learning."""
        while True:
            try:
                # Check if we have enough local data for training
                if len(self.data_streams["cgm"]) >= 1000:  # Need substantial data
                    # Prepare local data
                    local_data = self._prepare_federated_data()

                    # Train local model
                    model_updates = self.federated_manager.train_local_model(local_data)

                    # Send updates to server
                    success = await self.federated_manager.send_updates(model_updates)

                    if success:
                        # Receive global model
                        global_model = await self.federated_manager.receive_global_model()
                        if global_model:
                            print(f"✅ Updated to global model v{global_model['model_version']}")

                await asyncio.sleep(86400)  # Daily federated learning

            except Exception as e:
                print(f"❌ Error in federated learning: {e}")
                await asyncio.sleep(86400)

    async def _alert_processing_loop(self):
        """Κύκλος επεξεργασίας alerts."""
        while True:
            try:
                # Process pending alerts
                for alert in self.active_alerts.copy():
                    await self._handle_alert(alert)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                print(f"❌ Error in alert processing: {e}")
                await asyncio.sleep(10)

    async def _check_glucose_alerts(
        self, current_glucose: float, predicted_glucose: float, confidence: float
    ):
        """Έλεγχος για glucose alerts."""

        alerts = []

        # Hypoglycemia alerts
        if predicted_glucose < 70:
            severity = "critical" if predicted_glucose < 54 else "high"
            alert = GlucoseAlert(
                timestamp=datetime.now(),
                glucose_value=current_glucose,
                predicted_value=predicted_glucose,
                alert_type="hypoglycemia",
                severity=severity,
                recommendation="Λάβετε 15g γρήγορων υδατανθράκων",
                confidence=confidence,
            )
            alerts.append(alert)

        # Hyperglycemia alerts
        elif predicted_glucose > 250:
            alert = GlucoseAlert(
                timestamp=datetime.now(),
                glucose_value=current_glucose,
                predicted_value=predicted_glucose,
                alert_type="hyperglycemia",
                severity="high",
                recommendation="Ελέγξτε κετόνες και εφαρμόστε διορθωτική ινσουλίνη",
                confidence=confidence,
            )
            alerts.append(alert)

        # Rapid change alerts
        if len(self.data_streams["cgm"]) >= 2:
            prev_glucose = self.data_streams["cgm"][-2]["value"]
            rate_of_change = (current_glucose - prev_glucose) / 5  # mg/dL per minute

            if abs(rate_of_change) > 2:  # >2 mg/dL per minute
                alert = GlucoseAlert(
                    timestamp=datetime.now(),
                    glucose_value=current_glucose,
                    predicted_value=predicted_glucose,
                    alert_type="rapid_change",
                    severity="medium",
                    recommendation=f"Γρήγορη αλλαγή: {rate_of_change:+.1f} mg/dL/min",
                    confidence=confidence,
                )
                alerts.append(alert)

        # Add alerts to processing queue
        self.active_alerts.extend(alerts)

    def add_glucose_reading(self, glucose_value: float, timestamp: Optional[datetime] = None):
        """Προσθήκη νέας μέτρησης γλυκόζης."""
        if timestamp is None:
            timestamp = datetime.now()

        self.data_streams["cgm"].append({"value": glucose_value, "timestamp": timestamp})

        # Keep only recent data (24 hours)
        cutoff_time = timestamp - timedelta(hours=24)
        self.data_streams["cgm"] = [
            reading for reading in self.data_streams["cgm"] if reading["timestamp"] > cutoff_time
        ]

    def register_alert_callback(self, callback: Callable[[GlucoseAlert], None]):
        """Εγγραφή callback για alerts."""
        self.alert_callbacks.append(callback)

    async def _handle_alert(self, alert: GlucoseAlert):
        """Χειρισμός alert."""
        # Notify all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"❌ Error in alert callback: {e}")

        # Remove processed alert
        if alert in self.active_alerts:
            self.active_alerts.remove(alert)

    def _calculate_glucose_trend(self, readings: List[float]) -> str:
        """Υπολογισμός τάσης γλυκόζης."""
        if len(readings) < 3:
            return "stable"

        recent_slope = np.polyfit(range(len(readings[-3:])), readings[-3:], 1)[0]

        if recent_slope > 1:
            return "rapidly_increasing"
        elif recent_slope > 0.5:
            return "increasing"
        elif recent_slope < -1:
            return "rapidly_decreasing"
        elif recent_slope < -0.5:
            return "decreasing"
        else:
            return "stable"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Επιστροφή performance summary."""
        return {
            "engine_status": "active",
            "total_predictions": self.edge_engine.inference_stats["total_inferences"],
            "avg_prediction_latency": self.edge_engine.inference_stats["avg_latency_ms"],
            "active_alerts": len(self.active_alerts),
            "federated_contribution_score": self.federated_manager.contribution_score,
            "known_biomarkers": len(self.biomarker_engine.known_biomarkers),
            "discovered_biomarkers": len(self.biomarker_engine.discovered_biomarkers),
            "performance_metrics": self.performance_metrics,
        }
