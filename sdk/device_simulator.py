"""
🎮 Device Simulator - Virtual CGM/Pump for development & demos
==============================================================

full simulator that behaves exactly like real devices!
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Tuple, Union
import json


class VirtualPatient:
    """Realistic virtual patient model."""

    def __init__(
        self,
        patient_id: str = "virtual_001",
        age: int = 35,
        weight: float = 70,
        diabetes_type: str = "T1D",
        insulin_sensitivity: float = 50,  # mg/dL per unit
        carb_ratio: float = 10,
    ):  # grams per unit

        self.patient_id = patient_id
        self.age = age
        self.weight = weight
        self.diabetes_type = diabetes_type
        self.insulin_sensitivity = insulin_sensitivity
        self.carb_ratio = carb_ratio

        # Physiological state
        self.current_glucose = 120.0
        self.insulin_on_board = 0.0
        self.carbs_on_board = 0.0
        self.stress_level = 0.5  # 0-1
        self.activity_level = 0.3  # 0-1

        # History
        self.glucose_history = []
        self.insulin_history = []
        self.meal_history = []

    def update_glucose(self, minutes_elapsed: float = 5) -> float:
        """Update glucose based on all factors."""
        # Base metabolic rate
        liver_glucose = 1.5 * (minutes_elapsed / 60)  # mg/dL/hour

        # Insulin effect
        insulin_effect = self.insulin_on_board * self.insulin_sensitivity * (minutes_elapsed / 60)

        # Carb effect
        carb_effect = (self.carbs_on_board / self.carb_ratio) * 4 * (minutes_elapsed / 60)

        # Activity effect
        activity_effect = self.activity_level * 20 * (minutes_elapsed / 60)

        # Stress effect
        stress_effect = self.stress_level * 15 * (minutes_elapsed / 60)

        # Dawn phenomenon (if morning)
        hour = datetime.now().hour
        dawn_effect = 0
        if 4 <= hour <= 8:
            dawn_effect = 10 * (minutes_elapsed / 60)

        # Calculate new glucose
        glucose_change = (
            liver_glucose
            + carb_effect
            + stress_effect
            + dawn_effect
            - insulin_effect
            - activity_effect
        )

        self.current_glucose += glucose_change

        # Add realistic noise
        self.current_glucose += np.random.normal(0, 2)

        # Clamp to realistic range
        self.current_glucose = max(40, min(400, self.current_glucose))

        # Update IOB and COB decay
        self.insulin_on_board *= 0.95 ** (minutes_elapsed / 5)
        self.carbs_on_board *= 0.90 ** (minutes_elapsed / 5)

        # Store history
        self.glucose_history.append(
            {
                "timestamp": datetime.now(),
                "glucose": self.current_glucose,
                "iob": self.insulin_on_board,
                "cob": self.carbs_on_board,
            }
        )

        return self.current_glucose

    def inject_insulin(self, units: float, insulin_type: str = "rapid"):
        """Simulate insulin injection."""
        if insulin_type == "rapid":
            self.insulin_on_board += units
        elif insulin_type == "basal":
            self.insulin_on_board += units * 0.1  # Slower effect

        self.insulin_history.append(
            {"timestamp": datetime.now(), "units": units, "type": insulin_type}
        )

    def eat_meal(self, carbs: float, glycemic_index: str = "medium"):
        """Simulate meal consumption."""
        gi_factor = {"low": 0.5, "medium": 1.0, "high": 1.5}
        self.carbs_on_board += carbs * gi_factor.get(glycemic_index, 1.0)

        self.meal_history.append(
            {"timestamp": datetime.now(), "carbs": carbs, "gi": glycemic_index}
        )

    def set_activity(self, level: float):
        """Set activity level (0-1)."""
        self.activity_level = max(0, min(1, level))

    def set_stress(self, level: float):
        """Set stress level (0-1)."""
        self.stress_level = max(0, min(1, level))


class VirtualCGM:
    """Virtual CGM that behaves like real Dexcom/Freestyle."""

    def __init__(
        self,
        device_id: str = "VIRTUAL_CGM_001",
        manufacturer: str = "dexcom",
        model: str = "G6",
        patient: Optional[VirtualPatient] = None,
    ):

        self.device_id = device_id
        self.manufacturer = manufacturer
        self.model = model
        self.patient = patient or VirtualPatient()

        # Device characteristics
        self.accuracy_mard = 9.0  # % Mean Absolute Relative Difference
        self.warm_up_time = 120  # minutes
        self.sensor_life = 10 * 24 * 60  # 10 days in minutes

        # State
        self.is_connected = False
        self.sensor_start_time = None
        self.last_calibration = None
        self.battery_level = 100

    async def connect(self) -> bool:
        """Simulate device connection."""
        print(f"🔗 Connecting to Virtual {self.manufacturer.upper()} {self.model}...")
        await asyncio.sleep(1)  # Simulate connection delay

        self.is_connected = True
        self.sensor_start_time = datetime.now()
        print(f"✅ Virtual CGM connected! ID: {self.device_id}")

        return True

    def get_glucose_reading(self) -> Dict:
        """Get current glucose reading with CGM characteristics."""
        if not self.is_connected:
            raise Exception("CGM not connected")

        # Get true glucose from patient
        true_glucose = self.patient.current_glucose

        # Add CGM-specific error
        error_percent = np.random.normal(0, self.accuracy_mard / 100)
        measured_glucose = true_glucose * (1 + error_percent)

        # Add lag (CGM measures interstitial, not blood)
        lag_minutes = np.random.uniform(5, 15)

        # Calculate trend
        if len(self.patient.glucose_history) >= 3:
            recent = [h["glucose"] for h in self.patient.glucose_history[-3:]]
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            trend_arrow = self._get_trend_arrow(trend)
        else:
            trend_arrow = "→"

        reading = {
            "timestamp": datetime.now(),
            "glucose": round(measured_glucose),
            "trend": trend_arrow,
            "sensor_age_hours": self._get_sensor_age_hours(),
            "battery": self.battery_level,
            "device_id": self.device_id,
            "raw_value": true_glucose,  # For debugging
            "lag_minutes": lag_minutes,
        }

        # Simulate battery drain
        self.battery_level -= 0.01

        return reading

    def _get_trend_arrow(self, trend_value: float) -> str:
        """Convert trend to arrow like real CGM."""
        if trend_value > 3:
            return "↑↑"  # Rising rapidly
        elif trend_value > 1:
            return "↑"  # Rising
        elif trend_value < -3:
            return "↓↓"  # Falling rapidly
        elif trend_value < -1:
            return "↓"  # Falling
        else:
            return "→"  # Stable

    def _get_sensor_age_hours(self) -> float:
        """Get sensor age in hours."""
        if not self.sensor_start_time:
            return 0
        delta = datetime.now() - self.sensor_start_time
        return delta.total_seconds() / 3600

    async def stream_glucose(self, callback, interval_seconds: int = 300):
        """Stream glucose readings like real CGM."""
        while self.is_connected:
            # Update patient physiology
            self.patient.update_glucose(interval_seconds / 60)

            # Get CGM reading
            reading = self.get_glucose_reading()

            # Send to callback
            await callback(reading)

            # Wait for next reading
            await asyncio.sleep(interval_seconds)


class VirtualInsulinPump:
    """Virtual insulin pump with realistic behavior."""

    def __init__(
        self,
        device_id: str = "VIRTUAL_PUMP_001",
        manufacturer: str = "omnipod",
        model: str = "DASH",
        patient: Optional[VirtualPatient] = None,
    ):

        self.device_id = device_id
        self.manufacturer = manufacturer
        self.model = model
        self.patient = patient or VirtualPatient()

        # Pump characteristics
        self.basal_rates = [1.0] * 24  # Units per hour for each hour
        self.reservoir_capacity = 200  # Units
        self.reservoir_level = 150  # Current units

        # State
        self.is_connected = False
        self.is_suspended = False
        self.temp_basal = None

    async def connect(self) -> bool:
        """Connect to virtual pump."""
        print(f"💉 Connecting to Virtual {self.manufacturer.upper()} {self.model}...")
        await asyncio.sleep(0.5)

        self.is_connected = True
        print(f"✅ Virtual Pump connected! ID: {self.device_id}")

        # Start basal delivery
        asyncio.create_task(self._deliver_basal())

        return True

    async def deliver_bolus(self, units: float, extended: bool = False) -> Dict:
        """Deliver insulin bolus."""
        if not self.is_connected:
            raise Exception("Pump not connected")

        if units > self.reservoir_level:
            raise Exception("Insufficient insulin in reservoir")

        print(f"💉 Delivering {units}U bolus...")

        if extended:
            # Extended bolus over 2 hours
            for i in range(8):  # 8 x 15 min = 2 hours
                self.patient.inject_insulin(units / 8, "rapid")
                self.reservoir_level -= units / 8
                await asyncio.sleep(0.1)  # Simulated, would be 900 in real
        else:
            # Immediate bolus
            self.patient.inject_insulin(units, "rapid")
            self.reservoir_level -= units

        return {
            "timestamp": datetime.now(),
            "units_delivered": units,
            "type": "extended" if extended else "normal",
            "reservoir_remaining": self.reservoir_level,
        }

    async def _deliver_basal(self):
        """Background basal delivery."""
        while self.is_connected:
            if not self.is_suspended:
                hour = datetime.now().hour
                rate = self.basal_rates[hour]

                # Deliver in micro-boluses every 5 minutes
                micro_bolus = rate / 12  # 12 x 5min = 1 hour
                self.patient.inject_insulin(micro_bolus, "basal")
                self.reservoir_level -= micro_bolus

            await asyncio.sleep(300)  # 5 minutes


class DeviceSimulationPlatform:
    """Complete simulation platform για demos & development."""

    def __init__(self):
        self.patients: Dict[str, VirtualPatient] = {}
        self.cgms: Dict[str, VirtualCGM] = {}
        self.pumps: Dict[str, VirtualInsulinPump] = {}

    def create_scenario(
        self, scenario_name: str
    ) -> Tuple[VirtualPatient, VirtualCGM, VirtualInsulinPump]:
        """Create predefined scenarios."""

        scenarios = {
            "stable_adult": {
                "age": 35,
                "weight": 75,
                "insulin_sensitivity": 50,
                "carb_ratio": 10,
                "initial_glucose": 120,
            },
            "brittle_teen": {
                "age": 16,
                "weight": 60,
                "insulin_sensitivity": 65,
                "carb_ratio": 8,
                "initial_glucose": 180,
            },
            "dawn_phenomenon": {
                "age": 45,
                "weight": 80,
                "insulin_sensitivity": 40,
                "carb_ratio": 12,
                "initial_glucose": 110,
            },
        }

        config = scenarios.get(scenario_name, scenarios["stable_adult"])

        # Create patient
        initial_glucose = config.pop("initial_glucose", 120)
        patient = VirtualPatient(patient_id=f"scenario_{scenario_name}", **config)
        patient.current_glucose = initial_glucose

        # Create devices
        cgm = VirtualCGM(patient=patient)
        pump = VirtualInsulinPump(patient=patient)

        # Store
        self.patients[patient.patient_id] = patient
        self.cgms[cgm.device_id] = cgm
        self.pumps[pump.device_id] = pump

        return patient, cgm, pump

    async def run_day_simulation(self, patient_id: str):
        """Simulate a full day with meals, insulin, exercise."""
        patient = self.patients[patient_id]

        # Morning routine
        print("\n☀️ 07:00 - Morning")
        patient.current_glucose = 95

        # Breakfast
        print("🥞 08:00 - Breakfast (60g carbs)")
        patient.eat_meal(60, "medium")
        patient.inject_insulin(6, "rapid")  # 1:10 ratio

        # Work stress
        print("💼 10:00 - Work stress")
        patient.set_stress(0.7)

        # Lunch
        print("🥗 12:30 - Lunch (45g carbs)")
        patient.eat_meal(45, "low")
        patient.inject_insulin(4.5, "rapid")

        # Exercise
        print("🏃 17:00 - Exercise")
        patient.set_activity(0.8)
        patient.set_stress(0.2)

        # Dinner
        print("🍝 19:00 - Dinner (75g carbs)")
        patient.eat_meal(75, "high")
        patient.inject_insulin(7.5, "rapid")

        # Simulate the day
        for hour in range(24):
            for _ in range(12):  # Every 5 minutes
                patient.update_glucose(5)
                await asyncio.sleep(0.01)  # Fast simulation

            # Print hourly summary
            print(f"⏰ {hour:02d}:00 - Glucose: {patient.current_glucose:.0f} mg/dL")


# Integration helpers για SDK
def create_virtual_device(
    device_type: str, device_id: str = None
) -> Union[VirtualCGM, VirtualInsulinPump]:
    """Factory για virtual devices."""
    patient = VirtualPatient()

    if (
        "cgm" in device_type.lower()
        or "dexcom" in device_type.lower()
        or "freestyle" in device_type.lower()
    ):
        return VirtualCGM(
            device_id=device_id or f"VIRTUAL_CGM_{datetime.now().timestamp()}", patient=patient
        )
    elif "pump" in device_type.lower() or "omnipod" in device_type.lower():
        return VirtualInsulinPump(
            device_id=device_id or f"VIRTUAL_PUMP_{datetime.now().timestamp()}", patient=patient
        )
    else:
        raise ValueError(f"Unknown device type: {device_type}")


# Demo script
if __name__ == "__main__":

    async def demo():
        platform = DeviceSimulationPlatform()
        patient, cgm, pump = platform.create_scenario("brittle_teen")

        await cgm.connect()
        await pump.connect()

        # Get some readings
        for i in range(5):
            reading = cgm.get_glucose_reading()
            print(f"📊 Glucose: {reading['glucose']} mg/dL {reading['trend']}")
            patient.update_glucose(5)
            await asyncio.sleep(1)

        # Test bolus
        await pump.deliver_bolus(3.5)

        print("\nSimulation complete!")

    asyncio.run(demo())
