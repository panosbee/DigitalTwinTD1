"""
⚕️ Clinical Protocols & Guidelines για Digital Twin SDK
======================================================

Evidence-based protocols για healthcare providers.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ClinicalGuideline(Enum):
    """Διεθνή clinical guidelines."""
    ADA = "American Diabetes Association"
    EASD = "European Association for the Study of Diabetes"
    ISPAD = "International Society for Pediatric and Adolescent Diabetes"
    NICE = "National Institute for Health and Care Excellence"


@dataclass
class GlucoseTarget:
    """Στόχοι γλυκόζης ανά κατηγορία ασθενούς."""
    category: str
    pre_meal: Tuple[float, float]  # (min, max) mg/dL
    post_meal: Tuple[float, float]
    bedtime: Tuple[float, float]
    hba1c_target: float  # %


@dataclass 
class ClinicalAlert:
    """Clinical alert structure."""
    severity: str  # "low", "medium", "high", "critical"
    category: str
    message: str
    action_required: str
    evidence_base: str


class ClinicalProtocols:
    """
    Evidence-based clinical protocols για diabetes management.
    
    Βασισμένα σε ADA/EASD/ISPAD guidelines.
    """
    
    # Glucose targets by patient category
    GLUCOSE_TARGETS = {
        "adult_standard": GlucoseTarget(
            category="Adult Standard",
            pre_meal=(80, 130),
            post_meal=(80, 180),
            bedtime=(90, 150),
            hba1c_target=7.0
        ),
        "adult_tight": GlucoseTarget(
            category="Adult Tight Control",
            pre_meal=(70, 120),
            post_meal=(70, 160),
            bedtime=(80, 140),
            hba1c_target=6.5
        ),
        "pediatric": GlucoseTarget(
            category="Pediatric",
            pre_meal=(90, 130),
            post_meal=(90, 180),
            bedtime=(90, 150),
            hba1c_target=7.5
        ),
        "elderly": GlucoseTarget(
            category="Elderly/High Risk",
            pre_meal=(90, 150),
            post_meal=(100, 200),
            bedtime=(100, 180),
            hba1c_target=8.0
        ),
        "pregnancy": GlucoseTarget(
            category="Pregnancy",
            pre_meal=(60, 95),
            post_meal=(60, 140),
            bedtime=(60, 120),
            hba1c_target=6.0
        )
    }
    
    def __init__(self, guideline: ClinicalGuideline = ClinicalGuideline.ADA):
        self.guideline = guideline
        
    def get_glucose_targets(self, 
                           patient_category: str) -> GlucoseTarget:
        """
        Λήψη glucose targets για συγκεκριμένη κατηγορία ασθενούς.
        
        Args:
            patient_category: "adult_standard", "pediatric", etc.
            
        Returns:
            GlucoseTarget object
        """
        return self.GLUCOSE_TARGETS.get(
            patient_category, 
            self.GLUCOSE_TARGETS["adult_standard"]
        )
    
    def assess_glucose_control(self,
                              glucose_values: List[float],
                              patient_category: str) -> Dict:
        """
        Αξιολόγηση γλυκαιμικού ελέγχου βάσει guidelines.
        
        Returns:
            Dict με metrics και recommendations
        """
        targets = self.get_glucose_targets(patient_category)
        
        # Calculate metrics
        glucose_array = np.array(glucose_values)
        mean_glucose = np.mean(glucose_array)
        
        # Time in range
        in_range = np.sum((glucose_array >= 70) & (glucose_array <= 180))
        time_in_range = (in_range / len(glucose_array)) * 100
        
        # Hypoglycemia
        level_1_hypo = np.sum((glucose_array >= 54) & (glucose_array < 70))
        level_2_hypo = np.sum(glucose_array < 54)
        
        # Hyperglycemia  
        level_1_hyper = np.sum((glucose_array > 180) & (glucose_array <= 250))
        level_2_hyper = np.sum(glucose_array > 250)
        
        # Estimated HbA1c (Nathan formula)
        estimated_hba1c = (mean_glucose + 46.7) / 28.7
        
        assessment = {
            "mean_glucose": mean_glucose,
            "time_in_range": time_in_range,
            "time_below_range": (level_1_hypo + level_2_hypo) / len(glucose_array) * 100,
            "time_above_range": (level_1_hyper + level_2_hyper) / len(glucose_array) * 100,
            "estimated_hba1c": estimated_hba1c,
            "hypoglycemic_events": {
                "level_1": level_1_hypo,
                "level_2": level_2_hypo
            },
            "hyperglycemic_events": {
                "level_1": level_1_hyper,
                "level_2": level_2_hyper
            },
            "meets_targets": {
                "time_in_range": time_in_range >= 70,
                "hba1c": estimated_hba1c <= targets.hba1c_target
            }
        }
        
        return assessment
    
    def generate_clinical_alerts(self,
                                assessment: Dict,
                                patient_history: Optional[Dict] = None) -> List[ClinicalAlert]:
        """
        Δημιουργία clinical alerts βάσει assessment.
        
        Returns:
            List of ClinicalAlert objects
        """
        alerts = []
        
        # Severe hypoglycemia alert
        if assessment["hypoglycemic_events"]["level_2"] > 0:
            alerts.append(ClinicalAlert(
                severity="critical",
                category="hypoglycemia",
                message="Severe hypoglycemia detected (<54 mg/dL)",
                action_required="Immediate intervention required. Review insulin regimen.",
                evidence_base="ADA Standards of Care 2024"
            ))
        
        # Poor glycemic control
        if assessment["estimated_hba1c"] > 9.0:
            alerts.append(ClinicalAlert(
                severity="high",
                category="glycemic_control",
                message="Very poor glycemic control (HbA1c >9%)",
                action_required="Intensify therapy. Consider insulin adjustment or addition.",
                evidence_base="ADA/EASD Consensus Report"
            ))
        
        # Low time in range
        if assessment["time_in_range"] < 50:
            alerts.append(ClinicalAlert(
                severity="medium",
                category="time_in_range",
                message=f"Low time in range ({assessment['time_in_range']:.1f}%)",
                action_required="Review glucose patterns and adjust therapy",
                evidence_base="International Consensus on TIR"
            ))
        
        # Nocturnal hypoglycemia pattern
        # TODO: Add pattern detection
        
        return alerts
    
    def recommend_therapy_adjustments(self,
                                     assessment: Dict,
                                     current_therapy: Dict) -> List[Dict]:
        """
        Προτάσεις για προσαρμογή θεραπείας.
        
        Evidence-based recommendations.
        """
        recommendations = []
        
        # High fasting glucose
        if assessment["mean_glucose"] > 130:  # Pre-meal target
            recommendations.append({
                "type": "basal_adjustment",
                "rationale": "Elevated fasting glucose",
                "suggestion": "Increase basal insulin by 10%",
                "evidence": "Titration algorithms (INSIGHT trial)",
                "monitoring": "Check fasting glucose in 3 days"
            })
        
        # Postprandial excursions
        # TODO: Analyze meal patterns
        
        # Dawn phenomenon
        # TODO: Detect dawn phenomenon
        
        # Exercise adjustments
        # TODO: Analyze exercise patterns
        
        return recommendations
    
    def generate_patient_action_plan(self,
                                    assessment: Dict,
                                    alerts: List[ClinicalAlert]) -> Dict:
        """
        Δημιουργία action plan για τον ασθενή.
        
        Simple, actionable steps.
        """
        action_plan = {
            "immediate_actions": [],
            "daily_actions": [],
            "weekly_actions": [],
            "education_topics": []
        }
        
        # Immediate actions for critical alerts
        for alert in alerts:
            if alert.severity == "critical":
                if "hypoglycemia" in alert.category:
                    action_plan["immediate_actions"].append(
                        "Carry glucose tablets at all times"
                    )
                    action_plan["immediate_actions"].append(
                        "Review hypoglycemia treatment plan with family"
                    )
        
        # Daily monitoring
        if assessment["time_in_range"] < 70:
            action_plan["daily_actions"].append(
                "Check glucose before each meal and bedtime"
            )
            action_plan["daily_actions"].append(
                "Log meals and insulin doses"
            )
        
        # Weekly reviews
        action_plan["weekly_actions"].append(
            "Review glucose patterns every Sunday"
        )
        
        # Education based on assessment
        if assessment["hypoglycemic_events"]["level_1"] > 5:
            action_plan["education_topics"].append(
                "Hypoglycemia prevention and treatment"
            )
        
        return action_plan
    
    def calculate_insulin_sensitivity_factor(self,
                                           total_daily_dose: float,
                                           method: str = "1800_rule") -> float:
        """
        Υπολογισμός insulin sensitivity factor.
        
        Args:
            total_daily_dose: Total daily insulin
            method: "1800_rule" or "1500_rule"
            
        Returns:
            ISF in mg/dL per unit
        """
        if method == "1800_rule":
            return 1800 / total_daily_dose
        elif method == "1500_rule":
            return 1500 / total_daily_dose
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_carb_ratio(self,
                           total_daily_dose: float,
                           method: str = "500_rule") -> float:
        """
        Υπολογισμός carbohydrate ratio.
        
        Args:
            total_daily_dose: Total daily insulin
            method: "500_rule" or "450_rule"
            
        Returns:
            Grams of carbs per unit of insulin
        """
        if method == "500_rule":
            return 500 / total_daily_dose
        elif method == "450_rule":
            return 450 / total_daily_dose
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def assess_insulin_on_board(self,
                              last_bolus_time: float,
                              last_bolus_amount: float,
                              duration_hours: float = 4.0) -> float:
        """
        Υπολογισμός active insulin (IOB).
        
        Linear decay model.
        """
        if last_bolus_time >= duration_hours:
            return 0.0
        
        remaining_fraction = 1 - (last_bolus_time / duration_hours)
        return last_bolus_amount * remaining_fraction
    
    def recommend_sick_day_management(self,
                                    current_glucose: float,
                                    ketones_present: bool) -> Dict:
        """
        Sick day management protocol.
        
        Based on ISPAD guidelines.
        """
        protocol = {
            "check_ketones": True,
            "increase_monitoring": "Every 2-3 hours",
            "hydration": "Drink 250ml water every hour",
            "insulin_adjustment": "",
            "seek_medical_attention": False
        }
        
        if current_glucose > 250 and ketones_present:
            protocol["insulin_adjustment"] = "Give correction + 10-20% extra"
            protocol["seek_medical_attention"] = True
            protocol["urgency"] = "Within 2-4 hours if no improvement"
        
        return protocol
    
    def pediatric_specific_recommendations(self,
                                         age_years: int,
                                         assessment: Dict) -> List[str]:
        """
        Παιδιατρικές συστάσεις βάσει ISPAD.
        """
        recommendations = []
        
        if age_years < 6:
            recommendations.append(
                "Prioritize hypoglycemia prevention over tight control"
            )
            recommendations.append(
                "Target HbA1c <7.5% without excessive hypoglycemia"
            )
        
        if 6 <= age_years < 13:
            recommendations.append(
                "Target HbA1c <7.5%"
            )
            recommendations.append(
                "Focus on diabetes education and self-management skills"
            )
        
        if assessment["hypoglycemic_events"]["level_2"] > 0:
            recommendations.append(
                "Consider continuous glucose monitoring"
            )
            recommendations.append(
                "Review hypoglycemia awareness with family"
            )
        
        return recommendations


# Clinical decision support functions

def classify_glucose_pattern(glucose_values: List[float],
                           timestamps: List[str]) -> str:
    """
    Ταξινόμηση glucose patterns.
    
    Returns pattern type: "stable", "dawn_phenomenon", "rebound", etc.
    """
    # TODO: Implement pattern recognition
    return "stable"


def calculate_time_to_target(current_glucose: float,
                           target_glucose: float,
                           insulin_dose: float = 0) -> float:
    """
    Εκτίμηση χρόνου για επίτευξη στόχου.
    
    Returns minutes to reach target.
    """
    # Simplified model
    if insulin_dose > 0:
        # Assume 1U drops glucose by 50 mg/dL over 2 hours
        glucose_drop_rate = (insulin_dose * 50) / 120  # mg/dL per minute
        time_minutes = abs(current_glucose - target_glucose) / glucose_drop_rate
        return min(time_minutes, 240)  # Cap at 4 hours
    
    return 120  # Default 2 hours


def generate_clinical_summary(patient_data: Dict,
                            period_days: int = 14) -> str:
    """
    Δημιουργία clinical summary για healthcare provider.
    
    Concise, actionable summary.
    """
    summary = f"""
CLINICAL SUMMARY - {period_days} Day Report
=========================================

GLYCEMIC CONTROL:
- Mean Glucose: {patient_data.get('mean_glucose', 'N/A')} mg/dL
- Time in Range (70-180): {patient_data.get('time_in_range', 'N/A')}%
- Estimated HbA1c: {patient_data.get('estimated_hba1c', 'N/A')}%

HYPOGLYCEMIA:
- Level 1 (54-69 mg/dL): {patient_data.get('hypo_level_1', 0)} events
- Level 2 (<54 mg/dL): {patient_data.get('hypo_level_2', 0)} events

KEY PATTERNS:
- Dawn phenomenon: {'Yes' if patient_data.get('dawn_phenomenon') else 'No'}
- Post-meal excursions: {patient_data.get('postprandial_issue', 'Normal')}

RECOMMENDATIONS:
1. {patient_data.get('primary_recommendation', 'Continue current therapy')}
2. {patient_data.get('secondary_recommendation', 'Monitor glucose patterns')}

Next Review: {patient_data.get('next_review', 'In 3 months')}
"""
    
    return summary


# Export class για το testing script - αυτό χρειαζόταν!
class GlucoseTargets:
    """
    Helper class για glucose targets - wrapper για ClinicalProtocols.
    """
    
    @staticmethod
    def get_targets(patient_category: str, timing: str = "pre_meal") -> Tuple[float, float]:
        """Λήψη glucose targets για συγκεκριμένη κατηγορία και timing."""
        category_map = {
            "adult": "adult_standard",
            "pediatric": "pediatric", 
            "elderly": "elderly",
            "pregnancy": "pregnancy"
        }
        
        category_key = category_map.get(patient_category, "adult_standard")
        target = ClinicalProtocols.GLUCOSE_TARGETS[category_key]
        
        if timing == "pre_meal":
            return target.pre_meal
        elif timing == "post_meal":
            return target.post_meal
        elif timing == "bedtime":
            return target.bedtime
        else:
            return target.pre_meal
    
    @staticmethod 
    def check_glucose_alert(glucose_value: float) -> Dict:
        """Έλεγχος για glucose alerts."""
        if glucose_value < 54:
            return {"level": "critical", "message": "Severe hypoglycemia"}
        elif glucose_value < 70:
            return {"level": "high", "message": "Hypoglycemia"}
        elif glucose_value > 250:
            return {"level": "high", "message": "Severe hyperglycemia"}
        elif glucose_value > 180:
            return {"level": "medium", "message": "Hyperglycemia"}
        else:
            return {"level": "normal", "message": "Normal glucose"}


# Add missing method to ClinicalProtocols
def check_glucose_alert(glucose_value: float) -> Dict:
    """Static method για glucose alerts."""
    if glucose_value < 54:
        return {"level": "critical", "message": "Severe hypoglycemia"}
    elif glucose_value < 70:
        return {"level": "high", "message": "Hypoglycemia"}
    elif glucose_value > 250:
        return {"level": "high", "message": "Severe hyperglycemia"}
    elif glucose_value > 180:
        return {"level": "medium", "message": "Hyperglycemia"}
    else:
        return {"level": "normal", "message": "Normal glucose"}

# Attach to ClinicalProtocols class
ClinicalProtocols.check_glucose_alert = staticmethod(check_glucose_alert) 