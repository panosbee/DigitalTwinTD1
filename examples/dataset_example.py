"""
📊 Real Datasets Example with Digital Twin SDK
=============================================

Shows how to use real diabetes datasets from Kaggle, UCI, and other sources
with the Digital Twin SDK for research and development.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import DigitalTwinSDK, DiabetesDatasets, load_diabetes_data, list_available_datasets
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("📊 DIABETES DATASETS SHOWCASE")
    print("=" * 50)

    # 1. List available datasets
    print("\n1️⃣ Available Diabetes Datasets:")
    print("-" * 50)
    datasets_df = list_available_datasets()
    print(datasets_df[["name", "patients", "duration", "size"]])

    # 2. Load UCI Diabetes dataset (classic ML dataset)
    print("\n2️⃣ Loading UCI Pima Indians Diabetes Dataset:")
    print("-" * 50)

    try:
        uci_data = load_diabetes_data("uci_diabetes")
        print(f"✅ Loaded {len(uci_data)} samples")
        print(f"Features: {list(uci_data.columns)}")
        print(f"Diabetes prevalence: {uci_data['outcome'].mean()*100:.1f}%")
    except Exception as e:
        print(f"❌ Error loading UCI data: {e}")

    # 3. Generate synthetic CGM data for testing
    print("\n3️⃣ Generating Synthetic CGM Data:")
    print("-" * 50)

    cgm_data = load_diabetes_data("synthetic", n_patients=5, days=7)
    print(f"✅ Generated {len(cgm_data)} CGM readings")
    print(f"Patients: {cgm_data['patient_id'].nunique()}")
    print(f"Date range: {cgm_data['timestamp'].min()} to {cgm_data['timestamp'].max()}")

    # 4. Visualize CGM data
    print("\n4️⃣ Visualizing CGM Patterns:")
    print("-" * 50)

    # Plot first patient's week
    patient_1_data = cgm_data[cgm_data["patient_id"] == "synthetic_000"]

    plt.figure(figsize=(15, 6))
    plt.plot(patient_1_data["timestamp"], patient_1_data["cgm"], "b-", alpha=0.7)
    plt.axhspan(70, 180, alpha=0.2, color="green", label="Target Range")
    plt.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Hypo threshold")
    plt.axhline(y=180, color="orange", linestyle="--", alpha=0.5, label="Hyper threshold")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.title("Synthetic CGM Data - 1 Week Pattern")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 5. Use with Digital Twin SDK
    print("\n5️⃣ Using Real Data with Digital Twin SDK:")
    print("-" * 50)

    # Initialize SDK
    sdk = DigitalTwinSDK(mode="research")

    # Prepare dataset for training
    dataset_manager = DiabetesDatasets()

    # Get one patient's data
    patient_data = cgm_data[cgm_data["patient_id"] == "synthetic_001"].copy()

    # Prepare training windows
    X, y = dataset_manager.prepare_for_training(
        patient_data, target_col="cgm", lookback_hours=4, horizon_minutes=30
    )

    print(f"✅ Prepared training data:")
    print(f"   X shape: {X.shape} (samples, lookback_steps)")
    print(f"   y shape: {y.shape} (samples,)")
    print(f"   Ready for model training!")

    # 6. OpenAPS-style data (CGM + insulin + meals)
    print("\n6️⃣ Loading OpenAPS-style Dataset:")
    print("-" * 50)

    openaps_data = load_diabetes_data("openaps")
    print(f"✅ Loaded OpenAPS-style data: {len(openaps_data)} samples")
    print(f"Features: {list(openaps_data.columns)}")

    # Show meal and insulin events
    meal_events = openaps_data[openaps_data["carbs"] > 0]
    print(f"\nMeal events: {len(meal_events)}")
    print(f"Average carbs per meal: {meal_events['carbs'].mean():.1f}g")

    bolus_events = openaps_data[openaps_data["bolus"] > 0]
    print(f"Bolus events: {len(bolus_events)}")
    print(f"Average bolus: {bolus_events['bolus'].mean():.1f}U")

    # 7. Research example: Multi-patient analysis
    print("\n7️⃣ Multi-Patient Analysis:")
    print("-" * 50)

    # Calculate statistics per patient
    patient_stats = cgm_data.groupby("patient_id")["cgm"].agg(
        [
            "mean",
            "std",
            "min",
            "max",
            lambda x: ((x >= 70) & (x <= 180)).mean() * 100,  # Time in range
        ]
    )
    patient_stats.columns = ["Mean", "SD", "Min", "Max", "TIR %"]

    print("Patient Statistics:")
    print(patient_stats.round(1))

    # 8. Integration with clinical protocols
    print("\n8️⃣ Clinical Analysis with Real Data:")
    print("-" * 50)

    from sdk.clinical import ClinicalProtocols

    protocols = ClinicalProtocols()

    # Assess glucose control for one patient
    patient_glucose = patient_1_data["cgm"].values
    assessment = protocols.assess_glucose_control(
        patient_glucose, patient_category="adult_standard"
    )

    print(f"Clinical Assessment for Patient 1:")
    print(f"  Time in Range: {assessment['time_in_range']:.1f}%")
    print(f"  Estimated HbA1c: {assessment['estimated_hba1c']:.1f}%")
    print(f"  Hypoglycemic events: {assessment['hypoglycemic_events']['level_1']}")

    # 9. Save processed data for later use
    print("\n9️⃣ Saving Processed Data:")
    print("-" * 50)

    # Save to different formats
    output_dir = "./data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # CSV for general use
    cgm_data.to_csv(f"{output_dir}/cgm_data.csv", index=False)
    print(f"✅ Saved to CSV: {output_dir}/cgm_data.csv")

    # Parquet for efficient storage
    cgm_data.to_parquet(f"{output_dir}/cgm_data.parquet")
    print(f"✅ Saved to Parquet: {output_dir}/cgm_data.parquet")

    # 10. Tips for researchers
    print("\n🎯 Tips for Researchers:")
    print("-" * 50)
    print("• Use 'openaps' dataset for real-world CGM + pump data")
    print("• Use 'd1namo' for multi-modal data (CGM + activity)")
    print("• Use 'ohio_t1dm' for well-annotated research data")
    print("• Start with synthetic data for algorithm development")
    print("• Always respect patient privacy and data use agreements")

    print("\n✅ Dataset example completed!")
    print("🚀 Ready to revolutionize diabetes research!")


if __name__ == "__main__":
    main()
