Datasets
========

The Digital Twin T1D SDK provides access to various diabetes datasets for research and development.

Available Datasets
------------------

Use the SDK to list and load available datasets::

    from sdk.datasets import list_available_datasets, load_diabetes_data
    
    # List all available datasets
    print(list_available_datasets())
    
    # Load a specific dataset
    data = load_diabetes_data("uci_diabetes")

Synthetic Data Generation
-------------------------

Generate synthetic CGM data for testing::

    synthetic_data = load_diabetes_data("synthetic", n_patients=5, days=7)
    print(f"Generated {len(synthetic_data)} synthetic CGM readings.")

For more details, see the API documentation for the datasets module. 