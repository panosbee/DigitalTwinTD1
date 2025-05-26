#!/usr/bin/env python3
"""
🚀 Quick Start Guide για Digital Twin T1D SDK
===============================================

Γρήγορο παράδειγμα για να αρχίσεις σε 5 λεπτά!
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import DigitalTwinSDK, quick_predict, assess_glucose_risk


def main():
    """Quick start tutorial."""

    print("🎯 Digital Twin T1D SDK - Quick Start")
    print("=" * 50)

    # 1. Δημιουργία SDK instance
    print("\n1️⃣ Δημιουργία Digital Twin...")
    sdk = DigitalTwinSDK(mode="production")
    print("✅ SDK initialized!")

    # 2. Γρήγορη πρόβλεψη
    print("\n2️⃣ Γρήγορη glucose prediction...")
    glucose_history = [120, 125, 130, 135, 140]

    # Χρήση του helper function
    prediction = quick_predict(glucose_history)
    print(f"📊 Prediction: {prediction} mg/dL")

    # 3. Risk assessment
    print("\n3️⃣ Risk assessment...")
    risk = assess_glucose_risk(glucose_history[-1])
    print(f"⚠️  Risk level: {risk['level']} - {risk['message']}")

    # 4. Personalized recommendation
    print("\n4️⃣ Personalized recommendations...")
    recommendations = sdk.get_recommendations(
        {"current_glucose": glucose_history[-1], "trend": "rising", "meal_planned": False}
    )

    print("💡 Recommendations:")
    for rec in recommendations:
        print(f"   • {rec}")

    # 5. Σύνδεση με device (simulated)
    print("\n5️⃣ Device integration example...")
    try:
        from sdk.integrations import DeviceFactory

        # Λίστα supported devices
        devices = DeviceFactory.list_supported_devices()
        print(f"📱 Supported devices: {len(devices)}")
        print(f"   Examples: {', '.join(devices[:5])}...")

        # Demo device creation
        device = DeviceFactory.create_device("dexcom_g6", "demo_device")
        print(f"✅ Created device: {device.__class__.__name__}")

    except Exception as e:
        print(f"⚠️  Device integration: {e}")

    # 6. Load sample dataset
    print("\n6️⃣ Loading sample data...")
    try:
        from sdk.datasets import DatasetManager

        manager = DatasetManager()
        sample_data = manager.load_dataset("synthetic", size=100)
        print(f"📊 Loaded {len(sample_data)} synthetic samples")
        print(f"   Columns: {list(sample_data.columns)}")

    except Exception as e:
        print(f"⚠️  Dataset loading: {e}")

    # 7. Clinical protocols
    print("\n7️⃣ Clinical protocols...")
    try:
        from sdk.clinical import ClinicalProtocols, GlucoseTargets

        # Get glucose targets
        targets = GlucoseTargets.get_targets("adult", "pre_meal")
        print(f"🎯 Adult pre-meal targets: {targets[0]}-{targets[1]} mg/dL")

        # Clinical alert
        alert = ClinicalProtocols.check_glucose_alert(250)
        print(f"🚨 High glucose alert: {alert['message']}")

    except Exception as e:
        print(f"⚠️  Clinical protocols: {e}")

    print("\n" + "=" * 50)
    print("🎉 Quick start completed successfully!")
    print("\n📚 Next steps:")
    print("   • Check examples/sdk_showcase.py for advanced features")
    print("   • Read docs/ for complete documentation")
    print("   • Visit GitHub for latest updates")
    print("\n💝 Happy coding με το Digital Twin SDK!")


if __name__ == "__main__":
    main()
