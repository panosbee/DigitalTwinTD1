"""
🔌 Device Integrations για Digital Twin SDK
==========================================

Plug-and-play integrations για ΟΛΟΥΣ τους κατασκευαστές!
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import asyncio
import numpy as np
from dataclasses import dataclass
from enum import Enum


class DeviceStatus(Enum):
    """Κατάσταση συσκευής."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    PAIRING = "pairing"
    ERROR = "error"
    CALIBRATING = "calibrating"


@dataclass
class DeviceReading:
    """Γενική δομή για device readings."""

    timestamp: datetime
    value: float
    unit: str
    device_id: str
    raw_data: Optional[Dict] = None
    quality: float = 1.0  # 0-1 quality score


class DeviceIntegration(ABC):
    """
    Βασική κλάση για όλες τις device integrations.

    Κάθε κατασκευαστής μπορεί να την extend για να προσθέσει
    το δικό του device με 5 γραμμές κώδικα!
    """

    def __init__(self, device_id: str, config: Optional[Dict] = None):
        self.device_id = device_id
        self.config = config or {}
        self.status = DeviceStatus.DISCONNECTED
        self.last_reading = None
        self.callbacks = {}

    @abstractmethod
    async def connect(self) -> bool:
        """Σύνδεση με τη συσκευή."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Αποσύνδεση από τη συσκευή."""
        pass

    @abstractmethod
    async def get_reading(self) -> DeviceReading:
        """Λήψη τρέχουσας μέτρησης."""
        pass

    @abstractmethod
    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        """Λήψη ιστορικού μετρήσεων."""
        pass

    def on_reading(self, callback: Callable):
        """Register callback για νέες μετρήσεις."""
        self.callbacks["reading"] = callback

    def on_alert(self, callback: Callable):
        """Register callback για alerts."""
        self.callbacks["alert"] = callback

    async def start_streaming(self, interval_seconds: int = 300):
        """Έναρξη real-time streaming."""
        while self.status == DeviceStatus.CONNECTED:
            try:
                reading = await self.get_reading()
                if "reading" in self.callbacks:
                    await self.callbacks["reading"](reading)
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                print(f"Streaming error: {e}")
                break


# ===== CGM DEVICES =====


class CGMDevice(DeviceIntegration):
    """Βασική κλάση για CGM devices."""

    def __init__(self, device_id: str, config: Optional[Dict] = None):
        super().__init__(device_id, config)
        self.calibration_required = False
        self.sensor_age_days = 0

    async def calibrate(self, reference_glucose: float) -> bool:
        """Βαθμονόμηση CGM με reference τιμή."""
        print(f"Calibrating {self.device_id} with reference: {reference_glucose} mg/dL")
        self.calibration_required = False
        return True

    def get_trend_arrow(self) -> str:
        """Επιστροφή trend arrow."""
        # TODO: Calculate από recent readings
        return "→"  # Stable


class DexcomG6(CGMDevice):
    """
    Integration για Dexcom G6 CGM.

    Παράδειγμα για κατασκευαστές:
    ```python
    cgm = DexcomG6("serial_123")
    await cgm.connect()
    reading = await cgm.get_reading()
    ```
    """

    def __init__(self, device_id: str, config: Optional[Dict] = None):
        super().__init__(device_id, config)
        self.api_endpoint = self.config.get("api_endpoint", "https://api.dexcom.com")

    async def connect(self) -> bool:
        """Σύνδεση με Dexcom G6."""
        print(f"🔗 Connecting to Dexcom G6 ({self.device_id})...")
        # TODO: Implement actual Dexcom API connection
        self.status = DeviceStatus.CONNECTED
        print("✅ Dexcom G6 connected!")
        return True

    async def disconnect(self) -> bool:
        """Αποσύνδεση από Dexcom G6."""
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        """Λήψη τρέχουσας τιμής από Dexcom."""
        # TODO: Implement actual API call
        # Simulated reading
        return DeviceReading(
            timestamp=datetime.now(),
            value=120 + np.random.normal(0, 10),
            unit="mg/dL",
            device_id=self.device_id,
            quality=0.95,
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        """Λήψη ιστορικού από Dexcom."""
        # TODO: Implement actual API call
        readings = []
        for i in range(hours * 12):  # 5-min intervals
            readings.append(
                DeviceReading(
                    timestamp=datetime.now() - timedelta(minutes=i * 5),
                    value=120 + np.random.normal(0, 15),
                    unit="mg/dL",
                    device_id=self.device_id,
                )
            )
        return readings


class AbbottFreestyle(CGMDevice):
    """Integration για Abbott Freestyle Libre."""

    async def connect(self) -> bool:
        """Σύνδεση με Freestyle Libre."""
        print(f"🔗 Connecting to Freestyle Libre ({self.device_id})...")
        # TODO: Implement LibreLink API
        self.status = DeviceStatus.CONNECTED
        print("✅ Freestyle Libre connected!")
        return True

    async def disconnect(self) -> bool:
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        # TODO: Implement
        return DeviceReading(
            timestamp=datetime.now(),
            value=110 + np.random.normal(0, 8),
            unit="mg/dL",
            device_id=self.device_id,
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        # TODO: Implement
        return []


class MedtronicGuardian(CGMDevice):
    """Integration για Medtronic Guardian."""

    async def connect(self) -> bool:
        """Σύνδεση με Guardian."""
        print(f"🔗 Connecting to Medtronic Guardian ({self.device_id})...")
        # TODO: Implement CareLink API
        self.status = DeviceStatus.CONNECTED
        print("✅ Medtronic Guardian connected!")
        return True

    async def disconnect(self) -> bool:
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        # TODO: Implement
        return DeviceReading(
            timestamp=datetime.now(),
            value=115 + np.random.normal(0, 12),
            unit="mg/dL",
            device_id=self.device_id,
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        # TODO: Implement
        return []


class MockCGM(CGMDevice):
    """A mock CGM device for testing purposes."""

    async def connect(self) -> bool:
        print(f"🔩 MockCGM ({self.device_id}) connected.")
        self.status = DeviceStatus.CONNECTED
        return True

    async def disconnect(self) -> bool:
        print(f"🔩 MockCGM ({self.device_id}) disconnected.")
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        # Return a fairly stable, normal glucose reading
        return DeviceReading(
            timestamp=datetime.now(),
            value=110 + np.random.normal(0, 2),  # Low variance
            unit="mg/dL",
            device_id=self.device_id,
            quality=1.0,
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        readings = []
        for i in range(hours * 12):  # 5-min intervals
            readings.append(
                DeviceReading(
                    timestamp=datetime.now() - timedelta(minutes=i * 5),
                    value=110 + np.random.normal(0, 5),  # Some variance
                    unit="mg/dL",
                    device_id=self.device_id,
                )
            )
        return readings


# ===== INSULIN PUMPS =====


class InsulinPump(DeviceIntegration):
    """Βασική κλάση για insulin pumps."""

    def __init__(self, device_id: str, config: Optional[Dict] = None):
        super().__init__(device_id, config)
        self.basal_rate = 1.0  # Units/hour
        self.insulin_on_board = 0.0
        self.reservoir_units = 200.0

    async def deliver_bolus(self, units: float, meal: bool = False) -> bool:
        """Χορήγηση bolus insulin."""
        print(f"💉 Delivering {units}U bolus...")
        # TODO: Implement actual delivery
        self.insulin_on_board += units
        return True

    async def set_temp_basal(self, rate: float, duration_minutes: int) -> bool:
        """Ρύθμιση προσωρινού basal rate."""
        print(f"⚙️ Setting temp basal: {rate}U/hr for {duration_minutes} min")
        # TODO: Implement
        return True

    async def suspend_delivery(self) -> bool:
        """Αναστολή χορήγησης insulin."""
        print("⏸️ Suspending insulin delivery")
        # TODO: Implement
        return True

    def calculate_iob(self) -> float:
        """Υπολογισμός insulin on board."""
        # TODO: Implement decay calculation
        return self.insulin_on_board


class OmnipodDash(InsulinPump):
    """Integration για Omnipod DASH."""

    async def connect(self) -> bool:
        """Σύνδεση με Omnipod."""
        print(f"🔗 Connecting to Omnipod DASH ({self.device_id})...")
        # TODO: Implement Omnipod API
        self.status = DeviceStatus.CONNECTED
        print("✅ Omnipod DASH connected!")
        return True

    async def disconnect(self) -> bool:
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        """Λήψη τρέχουσας κατάστασης."""
        return DeviceReading(
            timestamp=datetime.now(),
            value=self.insulin_on_board,
            unit="Units",
            device_id=self.device_id,
            raw_data={
                "basal_rate": self.basal_rate,
                "reservoir": self.reservoir_units,
                "pod_age_hours": 48,
            },
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        # TODO: Implement
        return []


class TandemTslim(InsulinPump):
    """Integration για Tandem t:slim X2."""

    def __init__(self, device_id: str, config: Optional[Dict] = None):
        super().__init__(device_id, config)
        self.control_iq_enabled = True

    async def connect(self) -> bool:
        """Σύνδεση με t:slim."""
        print(f"🔗 Connecting to t:slim X2 ({self.device_id})...")
        # TODO: Implement t:connect API
        self.status = DeviceStatus.CONNECTED
        print("✅ t:slim X2 connected!")
        return True

    async def disconnect(self) -> bool:
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        return DeviceReading(
            timestamp=datetime.now(),
            value=self.insulin_on_board,
            unit="Units",
            device_id=self.device_id,
            raw_data={"basal_rate": self.basal_rate, "control_iq_active": self.control_iq_enabled},
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        # TODO: Implement
        return []


# ===== SMART WATCHES =====


class SmartWatch(DeviceIntegration):
    """Βασική κλάση για smartwatches."""

    def __init__(self, device_id: str, config: Optional[Dict] = None):
        super().__init__(device_id, config)
        self.heart_rate = 70
        self.steps = 0
        self.activity_level = "low"

    async def get_vitals(self) -> Dict[str, float]:
        """Λήψη vital signs."""
        return {
            "heart_rate": self.heart_rate,
            "steps": self.steps,
            "calories": self.steps * 0.04,
            "activity_minutes": self.steps // 100,
        }


class AppleWatch(SmartWatch):
    """Integration για Apple Watch."""

    async def connect(self) -> bool:
        """Σύνδεση με Apple Watch μέσω HealthKit."""
        print(f"🔗 Connecting to Apple Watch ({self.device_id})...")
        # TODO: Implement HealthKit integration
        self.status = DeviceStatus.CONNECTED
        print("✅ Apple Watch connected!")
        return True

    async def disconnect(self) -> bool:
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def get_reading(self) -> DeviceReading:
        """Λήψη activity data."""
        return DeviceReading(
            timestamp=datetime.now(),
            value=self.heart_rate,
            unit="bpm",
            device_id=self.device_id,
            raw_data=await self.get_vitals(),
        )

    async def get_history(self, hours: int = 24) -> List[DeviceReading]:
        # TODO: Implement
        return []


# ===== DEVICE FACTORY =====


class DeviceFactory:
    """
    Factory για εύκολη δημιουργία devices.

    Παράδειγμα:
    ```python
    device = DeviceFactory.create("dexcom_g6", "serial_123")
    await device.connect()
    ```
    """

    DEVICE_TYPES = {
        # CGMs
        "dexcom_g6": DexcomG6,
        "dexcom_g7": DexcomG6,  # Similar API
        "freestyle_libre": AbbottFreestyle,
        "freestyle_libre_2": AbbottFreestyle,
        "freestyle_libre_3": AbbottFreestyle,
        "guardian_3": MedtronicGuardian,
        "guardian_4": MedtronicGuardian,
        # Ensure MockCGM is defined before this line if it wasn't picked up by the previous diff part
        "mock_cgm": MockCGM,
        # Pumps
        "omnipod_dash": OmnipodDash,
        "omnipod_5": OmnipodDash,
        "tslim_x2": TandemTslim,
        "medtronic_770g": InsulinPump,  # Generic
        "medtronic_780g": InsulinPump,
        # Wearables
        "apple_watch": AppleWatch,
        "fitbit": SmartWatch,  # Generic
        "garmin": SmartWatch,
        "samsung_galaxy_watch": SmartWatch,
    }

    @classmethod
    def create(cls, device_type: str, device_id: str, config: Optional[Dict] = None):
        """Δημιουργία device instance."""
        device_type_lower = device_type.lower().replace(" ", "_")

        if device_type_lower not in cls.DEVICE_TYPES:
            raise ValueError(f"Unknown device type: {device_type}")

        device_class = cls.DEVICE_TYPES[device_type_lower]
        return device_class(device_id, config)

    @classmethod
    def list_supported_devices(cls) -> List[str]:
        """Λίστα υποστηριζόμενων devices."""
        return list(cls.DEVICE_TYPES.keys())

    @classmethod
    def create_device(
        cls, device_type: str, device_id: Optional[str] = None, config: Optional[Dict] = None
    ):
        """Alias για create method - για backward compatibility."""
        device_id = device_id or f"{device_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return cls.create(device_type, device_id, config)


# Export για εύκολη χρήση - αυτό χρειαζόταν για το test!
SUPPORTED_DEVICES = DeviceFactory.DEVICE_TYPES.copy()


# Helper functions
async def quick_connect(
    device_type: str, device_id: Optional[str] = None, config: Optional[Dict] = None
) -> DeviceIntegration:
    """Γρήγορη σύνδεση με device."""
    device = DeviceFactory.create_device(device_type, device_id, config)
    await device.connect()
    return device


def get_device_info(device_type: str) -> Dict[str, Any]:
    """Πληροφορίες για συγκεκριμένο device type."""
    if device_type not in SUPPORTED_DEVICES:
        return {"error": f"Device type '{device_type}' not supported"}

    device_class = SUPPORTED_DEVICES[device_type]
    return {
        "name": device_type,
        "class": device_class.__name__,
        "category": (
            "CGM"
            if issubclass(device_class, CGMDevice)
            else (
                "Pump"
                if issubclass(device_class, InsulinPump)
                else "Wearable" if issubclass(device_class, SmartWatch) else "Other"
            )
        ),
        "description": device_class.__doc__ or "No description available",
    }
