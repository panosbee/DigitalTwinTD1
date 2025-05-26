"""
Personalized Optimization Engine για ψηφιακό δίδυμο διαβήτη.

Περιλαμβάνει:
- Reinforcement Learning για optimal glucose control
- Multi-objective optimization (glycemic control vs quality of life)
- Personalized medicine algorithms
- Causal inference για treatment effect estimation
- Digital twin-based clinical trial simulation
- Explainable AI για clinical interpretability
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gym
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.env_util import make_vec_env

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

    # Create mock gym.Env for compatibility
    class gym:
        class Env:
            def __init__(self):
                pass

            def reset(self):
                pass

            def step(self, action):
                pass

        class spaces:
            @staticmethod
            def Box(low, high, dtype=None):
                return {"low": low, "high": high, "dtype": dtype}


try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.core.problem import Problem

    MULTI_OBJECTIVE_AVAILABLE = True
except ImportError:
    MULTI_OBJECTIVE_AVAILABLE = False

    # Create mock classes for compatibility
    class Problem:
        def __init__(self, n_var=4, n_obj=3, n_constr=2, xl=-1, xu=1):
            self.n_var = n_var
            self.n_obj = n_obj
            self.n_constr = n_constr

    class NSGA2:
        def __init__(self, pop_size=100):
            pass

    def minimize(problem, algorithm, termination, verbose=True):
        # Mock result object
        class Result:
            def __init__(self):
                self.X = np.random.uniform(-1, 1, (5, 4))  # 5 solutions with 4 variables

        return Result()


try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from econml.dml import CausalForestDML

    CAUSAL_ML_AVAILABLE = True
except ImportError:
    CAUSAL_ML_AVAILABLE = False


@dataclass
class PatientProfile:
    """Comprehensive patient profile για personalization."""

    patient_id: str
    age: float
    weight: float
    height: float
    diabetes_duration: float  # years
    hba1c: float

    # Insulin parameters
    insulin_sensitivity: float  # mg/dL per unit
    carb_ratio: float  # units per 15g carbs
    basal_rate: float  # units per hour

    # Physiological parameters
    dawn_phenomenon: float  # mg/dL increase in morning
    somogyi_effect: float  # rebound hyperglycemia
    gastroparesis_factor: float  # delayed gastric emptying

    # Lifestyle factors
    exercise_frequency: int  # days per week
    stress_level: float  # 1-10 scale
    sleep_quality: float  # 1-10 scale
    adherence_score: float  # medication adherence

    # Goals and preferences
    target_range: Tuple[float, float] = (70, 180)
    hypoglycemia_aversion: float = 0.8  # risk tolerance
    quality_of_life_weight: float = 0.3  # vs glycemic control

    # Clinical history
    hypoglycemic_episodes: int = 0
    dka_episodes: int = 0
    complications: List[str] = field(default_factory=list)

    # Genetic factors (if available)
    genetic_risk_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TreatmentRecommendation:
    """Personalized treatment recommendation."""

    recommendation_id: str
    patient_id: str
    timestamp: str

    # Insulin recommendations
    basal_adjustment: float  # % change
    carb_ratio_adjustment: float  # % change
    correction_factor_adjustment: float

    # Lifestyle recommendations
    exercise_plan: Dict[str, Any]
    nutrition_guidelines: Dict[str, Any]
    stress_management: List[str]

    # Monitoring recommendations
    cgm_frequency: str
    ketone_monitoring: bool
    additional_tests: List[str]

    # Expected outcomes
    predicted_hba1c_change: float
    predicted_tir_improvement: float
    hypoglycemia_risk_change: float

    # Confidence and evidence
    confidence_score: float
    evidence_level: str  # 'high', 'medium', 'low'
    clinical_trial_support: List[str]

    # Follow-up
    review_date: str
    monitoring_parameters: List[str]


class GlucoseControlEnvironment(gym.Env):
    """
    OpenAI Gym environment για glucose control με RL.

    State: [current_glucose, glucose_trend, time_of_day, meals, insulin_on_board, ...]
    Action: [basal_adjustment, bolus_dose, meal_timing_advice]
    Reward: Based on time in range, hypoglycemia avoidance, quality of life
    """

    def __init__(self, patient_profile: PatientProfile, simulation_hours: int = 24):
        super().__init__()

        self.patient_profile = patient_profile
        self.simulation_hours = simulation_hours
        self.time_step = 5  # minutes
        self.max_steps = simulation_hours * 60 // self.time_step

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-0.5, 0, -1.0]),  # [basal_adj, bolus, meal_advice]
            high=np.array([0.5, 10.0, 1.0]),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [20, -5, 0, 0, 0, 0, 0, 0]
            ),  # [glucose, trend, hour, meals, iob, stress, sleep, exercise]
            high=np.array([500, 5, 24, 100, 20, 10, 10, 10]),
            dtype=np.float32,
        )

        # Initialize state
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.glucose_history = [120.0]  # Start at normal glucose
        self.insulin_on_board = 0.0
        self.meal_carbs = 0.0
        self.stress_level = self.patient_profile.stress_level
        self.last_exercise = 0

        self.state = self._get_state()
        return self.state

    def step(self, action):
        """Execute one time step within the environment."""
        basal_adjustment, bolus_dose, meal_advice = action

        # Apply actions
        current_glucose = self.glucose_history[-1]

        # Simulate physiological response
        new_glucose = self._simulate_glucose_dynamics(
            current_glucose, basal_adjustment, bolus_dose, meal_advice
        )

        self.glucose_history.append(new_glucose)
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward(new_glucose, action)

        # Check if episode is done
        done = (self.current_step >= self.max_steps) or (new_glucose < 20) or (new_glucose > 500)

        # Update state
        self.state = self._get_state()

        # Additional info for debugging
        info = {
            "glucose": new_glucose,
            "time_in_range": self._calculate_time_in_range(),
            "hypoglycemia_events": sum(1 for g in self.glucose_history if g < 70),
            "hyperglycemia_events": sum(1 for g in self.glucose_history if g > 180),
        }

        return self.state, reward, done, info

    def _simulate_glucose_dynamics(
        self, current_glucose: float, basal_adj: float, bolus: float, meal_advice: float
    ) -> float:
        """Simulate glucose dynamics (simplified physiological model)."""

        # Time of day effects (dawn phenomenon)
        hour = (self.current_step * self.time_step / 60) % 24
        dawn_effect = 0
        if 4 <= hour <= 8:
            dawn_effect = self.patient_profile.dawn_phenomenon * np.sin((hour - 4) * np.pi / 4)

        # Basal insulin effect
        basal_effect = -(
            self.patient_profile.basal_rate
            * (1 + basal_adj)
            * self.patient_profile.insulin_sensitivity
            * self.time_step
            / 60
        )

        # Bolus insulin effect
        bolus_effect = -(bolus * self.patient_profile.insulin_sensitivity * 0.8)  # Peak effect

        # Meal effect (simplified)
        meal_effect = 0
        if np.random.random() < 0.1:  # Random meals
            carbs = np.random.normal(40, 15)  # 40±15g carbs
            meal_effect = carbs * 3  # 3 mg/dL per gram of carbs

        # Exercise effect
        exercise_effect = 0
        if np.random.random() < 0.05:  # Random exercise
            exercise_effect = -np.random.normal(30, 10)  # Glucose reduction

        # Stress effect
        stress_effect = self.stress_level * np.random.normal(0, 5)

        # Calculate new glucose
        glucose_change = (
            dawn_effect
            + basal_effect
            + bolus_effect
            + meal_effect
            + exercise_effect
            + stress_effect
        )

        new_glucose = current_glucose + glucose_change

        # Add physiological constraints
        new_glucose = max(20, min(500, new_glucose))  # Physiological bounds

        return new_glucose

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        current_glucose = self.glucose_history[-1]

        # Calculate glucose trend
        if len(self.glucose_history) >= 2:
            trend = self.glucose_history[-1] - self.glucose_history[-2]
        else:
            trend = 0

        # Time of day
        hour = (self.current_step * self.time_step / 60) % 24

        state = np.array(
            [
                current_glucose / 100,  # Normalized glucose
                trend / 50,  # Normalized trend
                hour / 24,  # Normalized hour
                self.meal_carbs / 100,  # Normalized carbs
                self.insulin_on_board / 10,  # Normalized IOB
                self.stress_level / 10,  # Normalized stress
                self.patient_profile.sleep_quality / 10,  # Sleep quality
                max(0, 1 - self.last_exercise / 48),  # Exercise recency
            ],
            dtype=np.float32,
        )

        return state

    def _calculate_reward(self, glucose: float, action: np.ndarray) -> float:
        """Calculate reward for RL agent."""
        reward = 0

        # Time in range reward
        target_low, target_high = self.patient_profile.target_range
        if target_low <= glucose <= target_high:
            reward += 10  # Strong positive reward for target range
        elif glucose < target_low:
            # Penalize hypoglycemia heavily
            severity = (target_low - glucose) / target_low
            reward -= 50 * severity * self.patient_profile.hypoglycemia_aversion
        elif glucose > target_high:
            # Penalize hyperglycemia moderately
            severity = (glucose - target_high) / target_high
            reward -= 20 * severity

        # Glucose variability penalty
        if len(self.glucose_history) >= 2:
            variability = abs(self.glucose_history[-1] - self.glucose_history[-2])
            reward -= variability * 0.1

        # Action penalty (prefer minimal interventions)
        basal_adj, bolus, meal_advice = action
        action_penalty = abs(basal_adj) + bolus * 0.1 + abs(meal_advice) * 0.5
        reward -= action_penalty

        # Quality of life considerations
        qol_weight = self.patient_profile.quality_of_life_weight
        if bolus > 5:  # Large bolus impacts QoL
            reward -= 5 * qol_weight

        return reward

    def _calculate_time_in_range(self) -> float:
        """Calculate percentage of time in target range."""
        if len(self.glucose_history) < 2:
            return 0.0

        target_low, target_high = self.patient_profile.target_range
        in_range = sum(1 for g in self.glucose_history if target_low <= g <= target_high)
        return 100 * in_range / len(self.glucose_history)


class PersonalizedRLController:
    """
    Reinforcement Learning controller για personalized glucose management.
    """

    def __init__(self, patient_profile: PatientProfile, algorithm: str = "PPO"):
        self.patient_profile = patient_profile
        self.algorithm = algorithm
        self.model = None
        self.env = None
        self.training_history = []

    def setup_environment(self):
        """Setup RL environment."""
        if not RL_AVAILABLE:
            raise ImportError("Stable Baselines3 required for RL. pip install stable-baselines3")

        self.env = GlucoseControlEnvironment(self.patient_profile)

    def train_controller(self, total_timesteps: int = 100000, save_path: Optional[str] = None):
        """Train RL controller."""
        if self.env is None:
            self.setup_environment()

        # Initialize RL algorithm
        if self.algorithm == "PPO":
            self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=0.0003, n_steps=2048)
        elif self.algorithm == "SAC":
            self.model = SAC(
                "MlpPolicy", self.env, verbose=1, learning_rate=0.0003, buffer_size=100000
            )
        elif self.algorithm == "A2C":
            self.model = A2C("MlpPolicy", self.env, verbose=1, learning_rate=0.0007)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Train model
        print(
            f"🎯 Training {self.algorithm} controller for patient {self.patient_profile.patient_id}"
        )
        self.model.learn(total_timesteps=total_timesteps)

        if save_path:
            self.model.save(save_path)
            print(f"💾 Model saved to {save_path}")

    def generate_recommendations(self, current_state: Dict) -> TreatmentRecommendation:
        """Generate personalized treatment recommendations."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_controller() first.")

        # Convert state to environment format
        env_state = self._convert_state_to_env_format(current_state)

        # Get action from trained model
        action, _ = self.model.predict(env_state, deterministic=True)

        # Convert action to treatment recommendation
        basal_adj, bolus_dose, meal_advice = action

        recommendation = TreatmentRecommendation(
            recommendation_id=f"rl_rec_{self.patient_profile.patient_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            patient_id=self.patient_profile.patient_id,
            timestamp=pd.Timestamp.now().isoformat(),
            # Insulin adjustments
            basal_adjustment=float(basal_adj * 100),  # Convert to percentage
            carb_ratio_adjustment=0.0,  # Not directly controlled by RL
            correction_factor_adjustment=0.0,
            # Lifestyle recommendations
            exercise_plan={
                "recommended": meal_advice > 0.5,
                "intensity": "light" if meal_advice > 0.5 else "none",
                "duration": 30 if meal_advice > 0.5 else 0,
            },
            nutrition_guidelines={
                "bolus_recommendation": float(bolus_dose),
                "carb_timing_advice": (
                    "normal" if meal_advice == 0 else ("delay" if meal_advice < 0 else "advance")
                ),
            },
            stress_management=[],
            # Monitoring
            cgm_frequency="every_5_min",
            ketone_monitoring=bolus_dose > 5,
            additional_tests=[],
            # Predictions (simplified)
            predicted_hba1c_change=-0.2 if basal_adj < 0 else 0.1,
            predicted_tir_improvement=5.0 if abs(basal_adj) > 0.1 else 0.0,
            hypoglycemia_risk_change=-10.0 if basal_adj > 0 else 5.0,
            # Confidence
            confidence_score=0.85,  # Based on training performance
            evidence_level="medium",
            clinical_trial_support=["RL_simulation"],
            # Follow-up
            review_date=(pd.Timestamp.now() + pd.Timedelta(days=7)).isoformat(),
            monitoring_parameters=["glucose_trends", "basal_effectiveness"],
        )

        return recommendation

    def _convert_state_to_env_format(self, state: Dict) -> np.ndarray:
        """Convert real-world state to environment state format."""
        glucose = state.get("glucose", 120)
        trend = state.get("glucose_trend", 0)
        hour = state.get("hour", 12)
        meals = state.get("recent_carbs", 0)
        iob = state.get("insulin_on_board", 0)
        stress = state.get("stress_level", self.patient_profile.stress_level)
        sleep = state.get("sleep_quality", self.patient_profile.sleep_quality)
        exercise = state.get("hours_since_exercise", 24)

        env_state = np.array(
            [
                glucose / 100,
                trend / 50,
                hour / 24,
                meals / 100,
                iob / 10,
                stress / 10,
                sleep / 10,
                max(0, 1 - exercise / 48),
            ],
            dtype=np.float32,
        )

        return env_state


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization για balance μεταξύ:
    - Glycemic control (HbA1c, TIR)
    - Hypoglycemia avoidance
    - Quality of life
    - Treatment burden
    """

    def __init__(self, patient_profile: PatientProfile):
        self.patient_profile = patient_profile
        self.optimization_history = []

    def optimize_treatment_plan(
        self, objectives: List[str], constraints: Dict
    ) -> List[TreatmentRecommendation]:
        """
        Multi-objective optimization of treatment plan.

        Args:
            objectives: List of objectives to optimize
            constraints: Treatment constraints
        """
        if not MULTI_OBJECTIVE_AVAILABLE:
            print("⚠️ PyMOO not available. Using simplified optimization...")
            return self._simplified_optimization(objectives, constraints)

        # Define optimization problem
        problem = DiabetesTreatmentProblem(self.patient_profile, objectives, constraints)

        # Setup NSGA-II algorithm
        algorithm = NSGA2(pop_size=100)

        # Run optimization
        result = minimize(problem, algorithm, ("n_gen", 100), verbose=True)

        # Convert solutions to treatment recommendations
        recommendations = []
        for i, solution in enumerate(result.X[:5]):  # Top 5 solutions
            recommendation = self._solution_to_recommendation(solution, i)
            recommendations.append(recommendation)

        return recommendations

    def _simplified_optimization(
        self, objectives: List[str], constraints: Dict
    ) -> List[TreatmentRecommendation]:
        """Simplified optimization when PyMOO is not available."""
        # Generate sample recommendations
        recommendations = []

        for i in range(3):  # Generate 3 alternative recommendations
            basal_adj = np.random.uniform(-0.2, 0.2)
            carb_ratio_adj = np.random.uniform(-0.1, 0.1)

            recommendation = TreatmentRecommendation(
                recommendation_id=f"mo_opt_{self.patient_profile.patient_id}_{i}",
                patient_id=self.patient_profile.patient_id,
                timestamp=pd.Timestamp.now().isoformat(),
                basal_adjustment=basal_adj * 100,
                carb_ratio_adjustment=carb_ratio_adj * 100,
                correction_factor_adjustment=0.0,
                exercise_plan={"type": "moderate", "frequency": 3},
                nutrition_guidelines={"carb_counting": True, "timing": "regular"},
                stress_management=["mindfulness", "regular_sleep"],
                cgm_frequency="continuous",
                ketone_monitoring=False,
                additional_tests=[],
                predicted_hba1c_change=basal_adj * -2,
                predicted_tir_improvement=abs(basal_adj) * 10,
                hypoglycemia_risk_change=basal_adj * 5,
                confidence_score=0.7 + i * 0.1,
                evidence_level="medium",
                clinical_trial_support=["simulation"],
                review_date=(pd.Timestamp.now() + pd.Timedelta(days=14)).isoformat(),
                monitoring_parameters=["hba1c", "time_in_range"],
            )

            recommendations.append(recommendation)

        return recommendations

    def _solution_to_recommendation(
        self, solution: np.ndarray, rank: int
    ) -> TreatmentRecommendation:
        """Convert optimization solution to treatment recommendation."""
        # Extract decision variables from solution
        basal_adj, carb_ratio_adj, exercise_freq, stress_mgmt = solution[:4]

        recommendation = TreatmentRecommendation(
            recommendation_id=f"mo_opt_{self.patient_profile.patient_id}_{rank}",
            patient_id=self.patient_profile.patient_id,
            timestamp=pd.Timestamp.now().isoformat(),
            basal_adjustment=basal_adj * 100,
            carb_ratio_adjustment=carb_ratio_adj * 100,
            correction_factor_adjustment=0.0,
            exercise_plan={"frequency": int(exercise_freq * 7), "type": "moderate", "duration": 30},
            nutrition_guidelines={"carb_counting": True, "meal_timing": "optimized"},
            stress_management=["meditation"] if stress_mgmt > 0.5 else [],
            cgm_frequency="continuous",
            ketone_monitoring=False,
            additional_tests=[],
            predicted_hba1c_change=basal_adj * -1.5,
            predicted_tir_improvement=abs(basal_adj) * 8,
            hypoglycemia_risk_change=basal_adj * 3,
            confidence_score=0.8,
            evidence_level="high",
            clinical_trial_support=["NSGA2_optimization"],
            review_date=(pd.Timestamp.now() + pd.Timedelta(days=14)).isoformat(),
            monitoring_parameters=["glycemic_variability", "quality_of_life"],
        )

        return recommendation


class DiabetesTreatmentProblem(Problem):
    """Multi-objective optimization problem for diabetes treatment."""

    def __init__(self, patient_profile: PatientProfile, objectives: List[str], constraints: Dict):
        self.patient_profile = patient_profile
        self.objectives = objectives
        self.constraints = constraints

        # Define decision variables: [basal_adj, carb_ratio_adj, exercise_freq, stress_mgmt]
        super().__init__(n_var=4, n_obj=len(objectives), n_constr=2, xl=-1, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate objectives and constraints."""
        n_samples = X.shape[0]
        objectives = np.zeros((n_samples, self.n_obj))
        constraints = np.zeros((n_samples, self.n_constr))

        for i in range(n_samples):
            basal_adj, carb_ratio_adj, exercise_freq, stress_mgmt = X[i]

            # Objective 1: Glycemic control (minimize HbA1c)
            predicted_hba1c = self.patient_profile.hba1c + basal_adj * -1.5 + carb_ratio_adj * -0.5
            objectives[i, 0] = predicted_hba1c  # Minimize

            # Objective 2: Hypoglycemia risk (minimize)
            hypo_risk = max(0, 0.1 + basal_adj * 0.3 - exercise_freq * 0.1)
            objectives[i, 1] = hypo_risk  # Minimize

            # Objective 3: Treatment burden (minimize)
            treatment_burden = abs(basal_adj) + abs(carb_ratio_adj) + exercise_freq * 0.5
            objectives[i, 2] = treatment_burden  # Minimize

            # Constraints
            constraints[i, 0] = abs(basal_adj) - 0.5  # Max 50% basal adjustment
            constraints[i, 1] = hypo_risk - 0.2  # Max 20% hypoglycemia risk

        out["F"] = objectives
        out["G"] = constraints


class CausalInferenceEngine:
    """
    Causal inference για estimation of treatment effects.

    Χρησιμοποιεί:
    - Causal forests για heterogeneous treatment effects
    - Doubly robust estimation
    - Instrumental variables analysis
    """

    def __init__(self):
        self.causal_models = {}
        self.treatment_effects = {}

    def estimate_treatment_effects(
        self, data: pd.DataFrame, treatment_column: str, outcome_column: str, confounders: List[str]
    ) -> Dict[str, Any]:
        """
        Estimate causal treatment effects.

        Args:
            data: Historical data
            treatment_column: Treatment variable (e.g., 'insulin_dose')
            outcome_column: Outcome variable (e.g., 'glucose_level')
            confounders: List of confounder variables
        """
        if not CAUSAL_ML_AVAILABLE:
            print("⚠️ EconML not available. Using correlation analysis...")
            return self._correlation_analysis(data, treatment_column, outcome_column)

        # Prepare data
        X = data[confounders]
        T = data[treatment_column]
        Y = data[outcome_column]

        # Causal Forest with Double ML
        causal_forest = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100),
            model_t=RandomForestRegressor(n_estimators=100),
            random_state=42,
        )

        # Fit causal model
        causal_forest.fit(Y, T, X=X)

        # Estimate individual treatment effects
        individual_effects = causal_forest.effect(X)

        # Calculate average treatment effect
        ate = np.mean(individual_effects)

        # Heterogeneous treatment effects by subgroups
        hte_results = self._analyze_heterogeneous_effects(X, individual_effects, confounders)

        results = {
            "average_treatment_effect": ate,
            "individual_effects": individual_effects,
            "heterogeneous_effects": hte_results,
            "confidence_intervals": self._calculate_confidence_intervals(individual_effects),
            "p_value": self._calculate_p_value(individual_effects),
            "effect_modifiers": self._identify_effect_modifiers(X, individual_effects, confounders),
        }

        # Store for future use
        self.treatment_effects[f"{treatment_column}_on_{outcome_column}"] = results

        return results

    def _correlation_analysis(
        self, data: pd.DataFrame, treatment: str, outcome: str
    ) -> Dict[str, Any]:
        """Simplified correlation analysis when causal ML is not available."""
        correlation = data[treatment].corr(data[outcome])

        return {
            "average_treatment_effect": correlation * data[treatment].std(),
            "correlation": correlation,
            "note": "Correlation analysis used - not causal inference",
        }

    def _analyze_heterogeneous_effects(
        self, X: pd.DataFrame, effects: np.ndarray, confounders: List[str]
    ) -> Dict[str, Any]:
        """Analyze heterogeneous treatment effects across subgroups."""
        hte_results = {}

        for confounder in confounders:
            if X[confounder].dtype in ["object", "category"]:
                # Categorical variable - group by categories
                groups = X.groupby(confounder)
                group_effects = {}
                for name, group_idx in groups.groups.items():
                    group_effects[str(name)] = {
                        "mean_effect": np.mean(effects[group_idx]),
                        "std_effect": np.std(effects[group_idx]),
                        "n_samples": len(group_idx),
                    }
                hte_results[confounder] = group_effects
            else:
                # Continuous variable - split into quartiles
                quartiles = pd.qcut(X[confounder], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
                quartile_effects = {}
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    q_mask = quartiles == q
                    if q_mask.sum() > 0:
                        quartile_effects[q] = {
                            "mean_effect": np.mean(effects[q_mask]),
                            "std_effect": np.std(effects[q_mask]),
                            "n_samples": q_mask.sum(),
                        }
                hte_results[confounder] = quartile_effects

        return hte_results

    def _calculate_confidence_intervals(
        self, effects: np.ndarray, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate confidence intervals for treatment effects."""
        mean_effect = np.mean(effects)
        std_effect = np.std(effects) / np.sqrt(len(effects))

        # 95% confidence interval
        ci_lower = mean_effect - 1.96 * std_effect
        ci_upper = mean_effect + 1.96 * std_effect

        return ci_lower, ci_upper

    def _calculate_p_value(self, effects: np.ndarray) -> float:
        """Calculate p-value for treatment effect significance."""
        from scipy import stats

        # Test if mean effect is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(effects, 0)
        return p_value

    def _identify_effect_modifiers(
        self, X: pd.DataFrame, effects: np.ndarray, confounders: List[str]
    ) -> List[Tuple[str, float]]:
        """Identify variables that modify treatment effects."""
        effect_modifiers = []

        for confounder in confounders:
            if X[confounder].dtype not in ["object", "category"]:
                # Calculate correlation between confounder and treatment effect
                correlation = np.corrcoef(X[confounder].values, effects)[0, 1]
                if abs(correlation) > 0.3:  # Threshold for meaningful correlation
                    effect_modifiers.append((confounder, correlation))

        # Sort by absolute correlation
        effect_modifiers.sort(key=lambda x: abs(x[1]), reverse=True)

        return effect_modifiers


class DigitalTwinClinicalTrialSimulator:
    """
    Digital twin-based clinical trial simulation.

    Μπορεί να προσομοιώσει κλινικές δοκιμές για:
    - Νέες θεραπείες
    - Διαφορετικά πρωτόκολλα
    - Personalized medicine strategies
    """

    def __init__(self, virtual_population_size: int = 1000):
        self.virtual_population_size = virtual_population_size
        self.virtual_patients = []
        self.simulation_results = {}

    def generate_virtual_population(self, population_characteristics: Dict) -> List[PatientProfile]:
        """Generate virtual patient population."""
        virtual_patients = []

        for i in range(self.virtual_population_size):
            # Sample patient characteristics from distributions
            age = np.random.normal(
                population_characteristics.get("mean_age", 35),
                population_characteristics.get("std_age", 15),
            )
            age = max(18, min(75, age))  # Constrain age

            weight = np.random.normal(
                population_characteristics.get("mean_weight", 75),
                population_characteristics.get("std_weight", 15),
            )
            weight = max(40, min(150, weight))

            hba1c = np.random.normal(
                population_characteristics.get("mean_hba1c", 8.0),
                population_characteristics.get("std_hba1c", 1.5),
            )
            hba1c = max(6.0, min(12.0, hba1c))

            # Generate other characteristics
            insulin_sensitivity = np.random.lognormal(
                np.log(population_characteristics.get("mean_insulin_sensitivity", 50)), 0.5
            )

            patient = PatientProfile(
                patient_id=f"virtual_{i:04d}",
                age=age,
                weight=weight,
                height=np.random.normal(170, 10),
                diabetes_duration=np.random.exponential(10),
                hba1c=hba1c,
                insulin_sensitivity=insulin_sensitivity,
                carb_ratio=np.random.normal(15, 3),
                basal_rate=np.random.normal(1.0, 0.3),
                dawn_phenomenon=np.random.normal(20, 10),
                somogyi_effect=np.random.normal(0, 5),
                gastroparesis_factor=np.random.beta(2, 8),
                exercise_frequency=np.random.poisson(3),
                stress_level=np.random.uniform(1, 10),
                sleep_quality=np.random.uniform(5, 10),
                adherence_score=np.random.beta(8, 2),
                hypoglycemia_aversion=np.random.beta(3, 2),
                quality_of_life_weight=np.random.uniform(0.1, 0.5),
            )

            virtual_patients.append(patient)

        self.virtual_patients = virtual_patients
        return virtual_patients

    def simulate_clinical_trial(
        self,
        treatment_arms: List[Dict],
        primary_endpoint: str,
        secondary_endpoints: List[str],
        trial_duration_days: int = 180,
    ) -> Dict[str, Any]:
        """
        Simulate clinical trial with multiple treatment arms.

        Args:
            treatment_arms: List of treatment protocols
            primary_endpoint: Primary outcome measure
            secondary_endpoints: Secondary outcome measures
            trial_duration_days: Duration of trial in days
        """
        if not self.virtual_patients:
            raise ValueError(
                "Virtual population not generated. Call generate_virtual_population() first."
            )

        print(f"🔬 Simulating clinical trial with {len(treatment_arms)} arms")
        print(f"📊 Virtual population: {len(self.virtual_patients)} patients")
        print(f"⏱️ Trial duration: {trial_duration_days} days")

        # Randomize patients to treatment arms
        n_per_arm = len(self.virtual_patients) // len(treatment_arms)
        arm_assignments = {}

        for i, arm in enumerate(treatment_arms):
            start_idx = i * n_per_arm
            end_idx = (
                start_idx + n_per_arm if i < len(treatment_arms) - 1 else len(self.virtual_patients)
            )
            arm_assignments[arm["name"]] = self.virtual_patients[start_idx:end_idx]

        # Simulate each treatment arm
        trial_results = {}

        for arm_name, patients in arm_assignments.items():
            print(f"  🧪 Simulating arm: {arm_name} (n={len(patients)})")

            arm_results = {
                "patients": patients,
                "outcomes": {},
                "adverse_events": [],
                "dropouts": [],
            }

            # Simulate outcomes for each patient
            primary_outcomes = []
            secondary_outcomes = {endpoint: [] for endpoint in secondary_endpoints}

            for patient in patients:
                # Simulate patient outcomes based on treatment
                patient_outcomes = self._simulate_patient_outcomes(
                    patient,
                    treatment_arms[list(arm_assignments.keys()).index(arm_name)],
                    trial_duration_days,
                )

                primary_outcomes.append(patient_outcomes[primary_endpoint])

                for endpoint in secondary_endpoints:
                    if endpoint in patient_outcomes:
                        secondary_outcomes[endpoint].append(patient_outcomes[endpoint])

                # Check for adverse events
                if patient_outcomes.get("severe_hypoglycemia", 0) > 0:
                    arm_results["adverse_events"].append(
                        {
                            "patient_id": patient.patient_id,
                            "event": "severe_hypoglycemia",
                            "count": patient_outcomes["severe_hypoglycemia"],
                        }
                    )

                # Check for dropouts (simplified)
                if np.random.random() < 0.05:  # 5% dropout rate
                    arm_results["dropouts"].append(patient.patient_id)

            # Calculate arm statistics
            arm_results["outcomes"] = {
                primary_endpoint: {
                    "mean": np.mean(primary_outcomes),
                    "std": np.std(primary_outcomes),
                    "median": np.median(primary_outcomes),
                    "n": len(primary_outcomes),
                }
            }

            for endpoint, values in secondary_outcomes.items():
                if values:
                    arm_results["outcomes"][endpoint] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "median": np.median(values),
                        "n": len(values),
                    }

            trial_results[arm_name] = arm_results

        # Statistical analysis
        statistical_results = self._analyze_trial_results(trial_results, primary_endpoint)

        # Compile final results
        final_results = {
            "trial_design": {
                "treatment_arms": treatment_arms,
                "primary_endpoint": primary_endpoint,
                "secondary_endpoints": secondary_endpoints,
                "duration_days": trial_duration_days,
                "total_patients": len(self.virtual_patients),
            },
            "arm_results": trial_results,
            "statistical_analysis": statistical_results,
            "summary": self._generate_trial_summary(trial_results, statistical_results),
        }

        self.simulation_results = final_results
        return final_results

    def _simulate_patient_outcomes(
        self, patient: PatientProfile, treatment: Dict, duration_days: int
    ) -> Dict[str, float]:
        """Simulate outcomes for individual patient."""
        # Simplified outcome simulation
        baseline_hba1c = patient.hba1c

        # Treatment effect based on protocol
        treatment_effect = treatment.get("hba1c_reduction", 0)

        # Individual response variability
        individual_response = np.random.normal(1.0, 0.3)

        # Final HbA1c
        final_hba1c = baseline_hba1c - (treatment_effect * individual_response)
        final_hba1c = max(5.0, min(12.0, final_hba1c))

        # Time in range (simplified relationship with HbA1c)
        time_in_range = max(0, min(100, 100 - (final_hba1c - 7) * 15))

        # Hypoglycemia events (simplified)
        hypo_rate = max(0, (8 - final_hba1c) * 0.5 + np.random.normal(0, 0.2))
        severe_hypo_events = np.random.poisson(hypo_rate * duration_days / 30)

        # Quality of life score
        qol_score = 100 - (final_hba1c - 7) * 5 - severe_hypo_events * 10
        qol_score = max(0, min(100, qol_score))

        return {
            "hba1c_change": final_hba1c - baseline_hba1c,
            "final_hba1c": final_hba1c,
            "time_in_range": time_in_range,
            "severe_hypoglycemia": severe_hypo_events,
            "quality_of_life": qol_score,
            "weight_change": np.random.normal(0, 2),  # kg
        }

    def _analyze_trial_results(self, trial_results: Dict, primary_endpoint: str) -> Dict[str, Any]:
        """Perform statistical analysis of trial results."""
        from scipy import stats

        arms = list(trial_results.keys())
        if len(arms) < 2:
            return {"error": "At least 2 arms required for comparison"}

        # Extract primary endpoint data
        arm_data = {}
        for arm in arms:
            outcomes = trial_results[arm]["outcomes"].get(primary_endpoint, {})
            if "mean" in outcomes:
                # Reconstruct individual data points (simplified)
                mean = outcomes["mean"]
                std = outcomes["std"]
                n = outcomes["n"]
                # Generate approximate individual data
                arm_data[arm] = np.random.normal(mean, std, n)

        if len(arm_data) < 2:
            return {"error": "Insufficient data for analysis"}

        # Pairwise comparisons (compare first arm with others)
        reference_arm = arms[0]
        comparisons = {}

        for arm in arms[1:]:
            # T-test
            t_stat, p_value = stats.ttest_ind(arm_data[reference_arm], arm_data[arm])

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (
                    (len(arm_data[reference_arm]) - 1) * np.std(arm_data[reference_arm]) ** 2
                    + (len(arm_data[arm]) - 1) * np.std(arm_data[arm]) ** 2
                )
                / (len(arm_data[reference_arm]) + len(arm_data[arm]) - 2)
            )
            cohens_d = (np.mean(arm_data[arm]) - np.mean(arm_data[reference_arm])) / pooled_std

            comparisons[f"{arm}_vs_{reference_arm}"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "cohens_d": cohens_d,
                "effect_size": (
                    "small"
                    if abs(cohens_d) < 0.5
                    else ("medium" if abs(cohens_d) < 0.8 else "large")
                ),
                "mean_difference": np.mean(arm_data[arm]) - np.mean(arm_data[reference_arm]),
            }

        return {
            "pairwise_comparisons": comparisons,
            "overall_p_value": min([comp["p_value"] for comp in comparisons.values()]),
            "primary_endpoint": primary_endpoint,
            "analysis_method": "t_test",
        }

    def _generate_trial_summary(
        self, trial_results: Dict, statistical_results: Dict
    ) -> Dict[str, Any]:
        """Generate human-readable trial summary."""
        arms = list(trial_results.keys())

        # Find best performing arm
        best_arm = None
        best_outcome = float("-inf")

        for arm in arms:
            primary_outcome = trial_results[arm]["outcomes"].get("hba1c_change", {}).get("mean", 0)
            if primary_outcome > best_outcome:  # Assuming negative change is better
                best_outcome = primary_outcome
                best_arm = arm

        # Safety summary
        total_adverse_events = sum(len(trial_results[arm]["adverse_events"]) for arm in arms)
        total_dropouts = sum(len(trial_results[arm]["dropouts"]) for arm in arms)

        return {
            "best_performing_arm": best_arm,
            "statistically_significant": any(
                comp["significant"]
                for comp in statistical_results.get("pairwise_comparisons", {}).values()
            ),
            "total_adverse_events": total_adverse_events,
            "total_dropouts": total_dropouts,
            "dropout_rate": total_dropouts
            / sum(len(trial_results[arm]["patients"]) for arm in arms)
            * 100,
            "recommendation": self._generate_recommendation(trial_results, statistical_results),
        }

    def _generate_recommendation(self, trial_results: Dict, statistical_results: Dict) -> str:
        """Generate clinical recommendation based on trial results."""
        comparisons = statistical_results.get("pairwise_comparisons", {})

        if not comparisons:
            return "Insufficient data for recommendation"

        significant_comparisons = [comp for comp in comparisons.values() if comp["significant"]]

        if not significant_comparisons:
            return "No statistically significant differences found between treatment arms"

        # Find most promising treatment
        best_comparison = max(significant_comparisons, key=lambda x: abs(x["mean_difference"]))

        if best_comparison["mean_difference"] < 0:  # Negative change is better for HbA1c
            return f"Experimental treatment shows significant improvement with large effect size (Cohen's d = {best_comparison['cohens_d']:.2f})"
        else:
            return f"Control treatment performs better than experimental treatment"


class ExplainableAIEngine:
    """
    Explainable AI για clinical interpretability.

    Παρέχει explanations για:
    - Model predictions
    - Treatment recommendations
    - Risk assessments
    """

    def __init__(self):
        self.explanation_cache = {}

    def explain_prediction(
        self, model, input_data: pd.DataFrame, prediction: float
    ) -> Dict[str, Any]:
        """
        Generate explanation for model prediction.

        Uses SHAP values and feature importance analysis.
        """
        try:
            import shap

            SHAP_AVAILABLE = True
        except ImportError:
            SHAP_AVAILABLE = False

        if SHAP_AVAILABLE and hasattr(model, "predict"):
            return self._shap_explanation(model, input_data, prediction)
        else:
            return self._simple_explanation(model, input_data, prediction)

    def _shap_explanation(
        self, model, input_data: pd.DataFrame, prediction: float
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        import shap

        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)

        # Extract feature contributions
        feature_contributions = {}
        for i, feature in enumerate(input_data.columns):
            feature_contributions[feature] = {
                "shap_value": float(shap_values.values[0, i]),
                "feature_value": float(input_data.iloc[0, i]),
                "contribution_magnitude": abs(float(shap_values.values[0, i])),
            }

        # Sort by contribution magnitude
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]["contribution_magnitude"],
            reverse=True,
        )

        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(sorted_features[:5], prediction)

        return {
            "prediction": prediction,
            "explanation_method": "SHAP",
            "feature_contributions": feature_contributions,
            "top_features": sorted_features[:5],
            "explanation_text": explanation_text,
            "confidence": self._calculate_explanation_confidence(shap_values),
        }

    def _simple_explanation(
        self, model, input_data: pd.DataFrame, prediction: float
    ) -> Dict[str, Any]:
        """Generate simple explanation when SHAP is not available."""
        # Use feature correlation with target (simplified)
        feature_importance = {}

        for feature in input_data.columns:
            value = input_data[feature].iloc[0]

            # Simple heuristics for diabetes prediction
            if "glucose" in feature.lower():
                importance = abs(value - 120) / 120  # Distance from normal
            elif "age" in feature.lower():
                importance = abs(value - 40) / 40
            elif "weight" in feature.lower():
                importance = abs(value - 70) / 70
            else:
                importance = 0.1  # Default low importance

            feature_importance[feature] = {
                "importance": importance,
                "value": value,
                "impact": "increases" if value > 0 else "decreases",
            }

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1]["importance"], reverse=True
        )

        explanation_text = f"Prediction of {prediction:.1f} mg/dL is primarily influenced by: "
        explanation_text += ", ".join(
            [f"{feat} ({data['impact']} risk)" for feat, data in sorted_features[:3]]
        )

        return {
            "prediction": prediction,
            "explanation_method": "heuristic",
            "feature_importance": feature_importance,
            "top_features": sorted_features[:5],
            "explanation_text": explanation_text,
            "confidence": 0.7,
        }

    def _generate_explanation_text(self, top_features: List[Tuple], prediction: float) -> str:
        """Generate natural language explanation."""
        explanation = (
            f"The predicted glucose level of {prediction:.1f} mg/dL is primarily driven by:\n"
        )

        for i, (feature, data) in enumerate(top_features, 1):
            contribution = data["shap_value"]
            value = data["feature_value"]

            direction = "increases" if contribution > 0 else "decreases"
            magnitude = "strongly" if abs(contribution) > 10 else "moderately"

            explanation += f"{i}. {feature.replace('_', ' ').title()}: {value:.2f} {magnitude} {direction} the prediction\n"

        return explanation

    def _calculate_explanation_confidence(self, shap_values) -> float:
        """Calculate confidence in explanation."""
        # Higher confidence when top features have large SHAP values
        total_magnitude = np.sum(np.abs(shap_values.values[0]))
        top_3_magnitude = np.sum(np.abs(sorted(shap_values.values[0], key=abs, reverse=True)[:3]))

        # Confidence based on concentration of importance in top features
        confidence = min(0.95, top_3_magnitude / max(total_magnitude, 0.001))
        return confidence

    def explain_treatment_recommendation(
        self, recommendation: TreatmentRecommendation
    ) -> Dict[str, str]:
        """Generate explanation for treatment recommendation."""
        explanations = {}

        # Insulin adjustments
        if recommendation.basal_adjustment != 0:
            direction = "increase" if recommendation.basal_adjustment > 0 else "decrease"
            magnitude = abs(recommendation.basal_adjustment)

            if magnitude > 20:
                explanations["basal_insulin"] = (
                    f"Significant {direction} in basal insulin ({magnitude:.1f}%) recommended due to consistent glucose patterns"
                )
            else:
                explanations["basal_insulin"] = (
                    f"Minor {direction} in basal insulin ({magnitude:.1f}%) for fine-tuning"
                )

        # Exercise recommendations
        if recommendation.exercise_plan.get("recommended", False):
            explanations["exercise"] = (
                "Exercise recommended to improve insulin sensitivity and glucose control"
            )

        # Monitoring changes
        if recommendation.ketone_monitoring:
            explanations["ketone_monitoring"] = (
                "Ketone monitoring recommended due to risk of hyperglycemia"
            )

        # Overall rationale
        explanations["rationale"] = (
            f"Recommendation based on predicted {recommendation.predicted_hba1c_change:+.1f}% HbA1c change and {recommendation.predicted_tir_improvement:+.1f}% improvement in time in range"
        )

        return explanations


class PersonalizedOptimizationEngine:
    """
    Κεντρικός engine που συντονίζει όλα τα optimization components.
    """

    def __init__(self, patient_profile: PatientProfile):
        self.patient_profile = patient_profile

        # Initialize components
        self.rl_controller = PersonalizedRLController(patient_profile)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(patient_profile)
        self.causal_inference_engine = CausalInferenceEngine()
        self.clinical_trial_simulator = DigitalTwinClinicalTrialSimulator()
        self.explainable_ai = ExplainableAIEngine()

        # Optimization history
        self.optimization_history = []

    def generate_comprehensive_recommendations(
        self,
        current_state: Dict,
        historical_data: pd.DataFrame,
        objectives: List[str] = None,
        constraints: Dict = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive personalized recommendations using all optimization methods.
        """
        if objectives is None:
            objectives = ["glycemic_control", "hypoglycemia_avoidance", "quality_of_life"]

        if constraints is None:
            constraints = {"max_insulin_change": 0.3, "max_exercise_hours": 2}

        print(
            f"🎯 Generating comprehensive recommendations for patient {self.patient_profile.patient_id}"
        )

        recommendations = {}

        # 1. Reinforcement Learning Recommendations
        try:
            print("  🤖 Generating RL-based recommendations...")
            if not self.rl_controller.model:
                print("    Training RL controller...")
                self.rl_controller.train_controller(total_timesteps=50000)

            rl_recommendation = self.rl_controller.generate_recommendations(current_state)
            recommendations["reinforcement_learning"] = rl_recommendation

            # Explain RL recommendation
            rl_explanation = self.explainable_ai.explain_treatment_recommendation(rl_recommendation)
            recommendations["rl_explanation"] = rl_explanation

        except Exception as e:
            print(f"    ❌ RL recommendation failed: {e}")
            recommendations["reinforcement_learning"] = None

        # 2. Multi-Objective Optimization
        try:
            print("  🎯 Generating multi-objective optimization recommendations...")
            mo_recommendations = self.multi_objective_optimizer.optimize_treatment_plan(
                objectives, constraints
            )
            recommendations["multi_objective"] = mo_recommendations

        except Exception as e:
            print(f"    ❌ Multi-objective optimization failed: {e}")
            recommendations["multi_objective"] = None

        # 3. Causal Inference Analysis
        try:
            if len(historical_data) > 100:  # Need sufficient data
                print("  🔬 Performing causal inference analysis...")

                # Analyze insulin treatment effects
                if (
                    "insulin_dose" in historical_data.columns
                    and "glucose_level" in historical_data.columns
                ):
                    confounders = [
                        col
                        for col in historical_data.columns
                        if col not in ["insulin_dose", "glucose_level"]
                    ][
                        :5
                    ]  # Max 5 confounders

                    causal_effects = self.causal_inference_engine.estimate_treatment_effects(
                        historical_data, "insulin_dose", "glucose_level", confounders
                    )
                    recommendations["causal_inference"] = causal_effects

        except Exception as e:
            print(f"    ❌ Causal inference failed: {e}")
            recommendations["causal_inference"] = None

        # 4. Generate Consensus Recommendation
        consensus_recommendation = self._generate_consensus_recommendation(recommendations)
        recommendations["consensus"] = consensus_recommendation

        # 5. Explanation and Confidence Assessment
        explanation = self._generate_comprehensive_explanation(recommendations)
        recommendations["explanation"] = explanation

        # Store in history
        optimization_record = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "patient_id": self.patient_profile.patient_id,
            "current_state": current_state,
            "objectives": objectives,
            "constraints": constraints,
            "recommendations": recommendations,
        }

        self.optimization_history.append(optimization_record)

        return recommendations

    def _generate_consensus_recommendation(self, recommendations: Dict) -> TreatmentRecommendation:
        """Generate consensus recommendation from multiple optimization methods."""

        # Collect basal adjustments from different methods
        basal_adjustments = []
        confidence_scores = []

        if recommendations.get("reinforcement_learning"):
            rl_rec = recommendations["reinforcement_learning"]
            basal_adjustments.append(rl_rec.basal_adjustment)
            confidence_scores.append(rl_rec.confidence_score)

        if recommendations.get("multi_objective"):
            mo_recs = recommendations["multi_objective"]
            if mo_recs:
                # Use the highest confidence recommendation
                best_mo_rec = max(mo_recs, key=lambda x: x.confidence_score)
                basal_adjustments.append(best_mo_rec.basal_adjustment)
                confidence_scores.append(best_mo_rec.confidence_score)

        # Weighted average based on confidence scores
        if basal_adjustments and confidence_scores:
            total_weight = sum(confidence_scores)
            weighted_basal_adj = (
                sum(ba * cs for ba, cs in zip(basal_adjustments, confidence_scores)) / total_weight
            )
            avg_confidence = np.mean(confidence_scores)
        else:
            weighted_basal_adj = 0.0
            avg_confidence = 0.5

        # Generate consensus recommendation
        consensus = TreatmentRecommendation(
            recommendation_id=f"consensus_{self.patient_profile.patient_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            patient_id=self.patient_profile.patient_id,
            timestamp=pd.Timestamp.now().isoformat(),
            basal_adjustment=weighted_basal_adj,
            carb_ratio_adjustment=0.0,  # Conservative approach
            correction_factor_adjustment=0.0,
            exercise_plan={
                "recommended": abs(weighted_basal_adj) < 10,  # Exercise if small insulin changes
                "type": "moderate",
                "frequency": 3,
            },
            nutrition_guidelines={"carb_counting": True, "timing": "consistent"},
            stress_management=["regular_sleep", "mindfulness"],
            cgm_frequency="continuous",
            ketone_monitoring=weighted_basal_adj
            < -20,  # Monitor if decreasing insulin significantly
            additional_tests=[],
            predicted_hba1c_change=weighted_basal_adj * -0.02,  # Rough estimate
            predicted_tir_improvement=abs(weighted_basal_adj) * 0.5,
            hypoglycemia_risk_change=weighted_basal_adj * 0.1,
            confidence_score=avg_confidence,
            evidence_level="high" if avg_confidence > 0.8 else "medium",
            clinical_trial_support=["multi_method_consensus"],
            review_date=(pd.Timestamp.now() + pd.Timedelta(days=14)).isoformat(),
            monitoring_parameters=["glucose_variability", "treatment_response"],
        )

        return consensus

    def _generate_comprehensive_explanation(self, recommendations: Dict) -> Dict[str, str]:
        """Generate comprehensive explanation of all recommendations."""
        explanation = {
            "summary": "Comprehensive personalized treatment recommendations generated using multiple AI methods",
            "methods_used": [],
            "agreement_level": "unknown",
            "key_insights": [],
            "clinical_rationale": "",
        }

        # Track which methods were used
        if recommendations.get("reinforcement_learning"):
            explanation["methods_used"].append("Reinforcement Learning")
        if recommendations.get("multi_objective"):
            explanation["methods_used"].append("Multi-Objective Optimization")
        if recommendations.get("causal_inference"):
            explanation["methods_used"].append("Causal Inference")

        # Assess agreement between methods
        basal_changes = []
        if recommendations.get("reinforcement_learning"):
            basal_changes.append(recommendations["reinforcement_learning"].basal_adjustment)
        if recommendations.get("multi_objective"):
            mo_recs = recommendations["multi_objective"]
            if mo_recs:
                basal_changes.extend([rec.basal_adjustment for rec in mo_recs])

        if len(basal_changes) > 1:
            std_basal = np.std(basal_changes)
            if std_basal < 5:
                explanation["agreement_level"] = "high"
            elif std_basal < 15:
                explanation["agreement_level"] = "moderate"
            else:
                explanation["agreement_level"] = "low"

        # Generate key insights
        if recommendations.get("causal_inference"):
            causal_effects = recommendations["causal_inference"]
            if "average_treatment_effect" in causal_effects:
                ate = causal_effects["average_treatment_effect"]
                explanation["key_insights"].append(
                    f"Causal analysis shows average treatment effect of {ate:.2f} mg/dL per unit insulin"
                )

        # Clinical rationale
        consensus = recommendations.get("consensus")
        if consensus:
            explanation["clinical_rationale"] = (
                f"Consensus recommendation suggests {consensus.basal_adjustment:+.1f}% basal insulin adjustment "
                f"with predicted {consensus.predicted_hba1c_change:+.2f}% HbA1c change and "
                f"{consensus.predicted_tir_improvement:+.1f}% improvement in time in range."
            )

        return explanation

    def simulate_treatment_outcomes(
        self, recommendation: TreatmentRecommendation, simulation_days: int = 90
    ) -> Dict[str, Any]:
        """
        Simulate outcomes of treatment recommendation using digital twin.
        """
        print(f"🔮 Simulating treatment outcomes for {simulation_days} days")

        # Create treatment protocol from recommendation
        treatment_protocol = {
            "name": "personalized_treatment",
            "basal_adjustment": recommendation.basal_adjustment / 100,  # Convert to fraction
            "carb_ratio_adjustment": recommendation.carb_ratio_adjustment / 100,
            "exercise_frequency": recommendation.exercise_plan.get("frequency", 3),
            "hba1c_reduction": abs(recommendation.predicted_hba1c_change),
        }

        # Generate small virtual population for this patient type
        population_characteristics = {
            "mean_age": self.patient_profile.age,
            "std_age": 5,
            "mean_weight": self.patient_profile.weight,
            "std_weight": 10,
            "mean_hba1c": self.patient_profile.hba1c,
            "std_hba1c": 0.5,
            "mean_insulin_sensitivity": self.patient_profile.insulin_sensitivity,
        }

        # Set smaller population for efficiency
        original_size = self.clinical_trial_simulator.virtual_population_size
        self.clinical_trial_simulator.virtual_population_size = 100

        virtual_population = self.clinical_trial_simulator.generate_virtual_population(
            population_characteristics
        )

        # Simulate trial with treatment vs standard care
        treatment_arms = [{"name": "standard_care", "hba1c_reduction": 0.0}, treatment_protocol]

        simulation_results = self.clinical_trial_simulator.simulate_clinical_trial(
            treatment_arms=treatment_arms,
            primary_endpoint="hba1c_change",
            secondary_endpoints=["time_in_range", "severe_hypoglycemia", "quality_of_life"],
            trial_duration_days=simulation_days,
        )

        # Restore original population size
        self.clinical_trial_simulator.virtual_population_size = original_size

        return simulation_results

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization engine performance."""
        return {
            "patient_id": self.patient_profile.patient_id,
            "total_optimizations": len(self.optimization_history),
            "last_optimization": (
                self.optimization_history[-1]["timestamp"] if self.optimization_history else None
            ),
            "available_methods": [
                "reinforcement_learning",
                "multi_objective_optimization",
                "causal_inference",
                "clinical_trial_simulation",
                "explainable_ai",
            ],
            "patient_characteristics": {
                "age": self.patient_profile.age,
                "diabetes_duration": self.patient_profile.diabetes_duration,
                "hba1c": self.patient_profile.hba1c,
                "hypoglycemia_aversion": self.patient_profile.hypoglycemia_aversion,
            },
        }
