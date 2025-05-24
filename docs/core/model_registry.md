# Model Registry

## 1. Introduction

### What is the Model Registry?

The Model Registry is a central component within the Digital Twin T1D library responsible for managing and providing access to various predictive models. It acts as a directory where model classes can be "registered" under a unique name, allowing them to be easily discovered and instantiated by other parts of the library, particularly the `DigitalTwin` class.

### Why is it used?

The primary purposes of the Model Registry are:

*   **Extensibility:** It allows developers to easily integrate custom-built models into the Digital Twin T1D ecosystem without modifying the core library code. By simply decorating a custom model class, it becomes available for use.
*   **Decoupling:** It decouples the `DigitalTwin` class (and other potential model consumers) from specific model implementations. The `DigitalTwin` only needs to know the registered name of a model (e.g., "lstm", "custom_model_v2") to load and use it, rather than having direct import dependencies on every possible model class.
*   **Simplified Configuration:** Users can specify which model to use via a string name in configurations or parameters, making it easier to switch between different models.
*   **Discovery:** Provides a mechanism to list all available models within the system.

## 2. Registering a Custom Model

To make a custom model available to the Digital Twin T1D library, you need to register it using the `@register_model` decorator.

### `@register_model("your_model_name")`

The `@register_model` decorator is imported from `digital_twin_t1d.core.model_registry`. You apply it directly above your model class definition.

*   **`model_name` (string):** This is a mandatory argument and defines the unique name under which your model will be registered. This name is used when requesting the model (e.g., in `DigitalTwin`'s `model_type` parameter).
*   **`overwrite=True` (boolean, optional):** If you attempt to register a model with a name that already exists, a `RegistrationError` will be raised by default. If you set `overwrite=True`, the registry will replace the existing model associated with that name with your new model class. Use this with caution.

### Code Example

Here's a simple example of a custom model class definition with the decorator:

```python
# my_custom_models/my_awesome_model.py
from digital_twin_t1d.core.model_registry import register_model
from digital_twin_t1d.core.twin import BaseModel # Assuming your model inherits from BaseModel
import pandas as pd
import numpy as np

@register_model("awesome_v1")
class MyAwesomeModel(BaseModel):
    """
    A custom 'awesome_v1' model.
    This model should implement the BaseModel interface.
    """
    def __init__(self, param1: int = 10, param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted = False
        # ... other initializations

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'MyAwesomeModel':
        # Your model training logic here
        print(f"Fitting AwesomeModel with X_shape={X.shape}, y_shape={y.shape}, param1={self.param1}, param2={self.param2}")
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MyAwesomeModel has not been fitted yet.")
        # Your prediction logic here
        print(f"Predicting with AwesomeModel for X_shape={X.shape}")
        # Example: return a dummy prediction based on the number of input rows
        return np.random.rand(len(X)) 

    def get_params(self) -> dict:
        return {"param1": self.param1, "param2": self.param2}

    def set_params(self, **params) -> 'MyAwesomeModel':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    # Potentially other BaseModel methods like save/load
```

## 3. Model Discovery and Usage

For a registered model to be available (e.g., for `DigitalTwin` to use it), its Python module must be imported *before* the model is requested. The act of importing the module typically executes the `@register_model` decorator, adding the model to the registry.

### Built-in Models

The Digital Twin T1D library includes several built-in models (like "lstm", "mechanistic"). These are typically made available when you import from the `digital_twin_t1d.models` package. For example, the `digital_twin_t1d.models.__init__.py` file imports its submodules (like `lstm.py`), which triggers the registration of any models defined within them.

```python
# This import ensures models like 'lstm' are registered
import digital_twin_t1d.models 
```

### Custom Models

To use your custom model, you need to ensure the Python module containing your custom model class is imported before you try to instantiate `DigitalTwin` with it.

```python
# main_script.py

# 1. Import your custom model's module to register it
import my_custom_models.my_awesome_model 

# 2. Now you can import and use DigitalTwin with your registered model
from digital_twin_t1d.core import DigitalTwin
from digital_twin_t1d.core.model_registry import list_available_models

# (Optional) Verify it's registered
print("Available models:", list_available_models()) 
# Expected output might include: {'awesome_v1': 'my_custom_models.my_awesome_model.MyAwesomeModel', ...}

# 3. Instantiate DigitalTwin with your custom model
try:
    twin = DigitalTwin(
        model_type="awesome_v1",  # Use the registered name
        model_params={"param1": 20, "param2": 0.75} 
    )
    print("DigitalTwin instantiated with awesome_v1 successfully!")
    # Now you can use the twin, e.g., twin.fit(X_train, y_train), twin.predict(X_test)
except Exception as e:
    print(f"Error instantiating DigitalTwin: {e}")

```

### Automatic Discovery with `discover_models`

The `digital_twin_t1d.core.model_registry` also provides a `discover_models(package_path: str)` function. This function can automatically import all modules within a specified package path, triggering the registration of any models defined therein. For example, if your custom models are organized under a package named `my_plugin_models`, you could call:

```python
from digital_twin_t1d.core.model_registry import discover_models, list_available_models

discover_models("my_plugin_models") 
# This would attempt to import all modules in my_plugin_models, registering them.

print(list_available_models())
```
This is particularly useful for plugin-based architectures where models might be contributed by different packages.

## 4. Listing Available Models

You can easily check which models are currently registered and available for use. The `list_available_models()` function from `digital_twin_t1d.core.model_registry` returns a dictionary where keys are the registered model names and values are their fully qualified class names.

```python
from digital_twin_t1d.core.model_registry import list_available_models

# Make sure models are imported/discovered first
import digital_twin_t1d.models # For built-in models
import my_custom_models.my_awesome_model # For your custom model

available_models = list_available_models()
print(available_models)
```

Example output:

```
{
    'lstm': 'models.lstm.LSTMModel',
    'mechanistic_uva_padova': 'models.mechanistic.MechanisticModel', 
    'awesome_v1': 'my_custom_models.my_awesome_model.MyAwesomeModel'
    # ... other registered models
}
```

## 5. Advanced Usage (Optional)

For more advanced scenarios, such as dynamic plugin management during runtime or for specific testing needs, the model registry offers a couple more utilities:

*   **`unregister_model(model_name: str) -> bool`:**
    Removes a model from the registry. Returns `True` if successful, `False` if the model was not found.
    ```python
    from digital_twin_t1d.core.model_registry import unregister_model
    unregistered = unregister_model("awesome_v1")
    print(f"Model 'awesome_v1' unregistered: {unregistered}")
    ```

*   **`is_model_registered(model_name: str) -> bool`:**
    Checks if a model with the given name is currently in the registry.
    ```python
    from digital_twin_t1d.core.model_registry import is_model_registered
    is_present = is_model_registered("lstm")
    print(f"Model 'lstm' is registered: {is_present}")
    ```

## 6. Best Practices

*   **Choose a Unique `model_name`:** Model names are global identifiers within the registry. Choose names that are descriptive and unlikely to clash with other models, especially if you plan to share or integrate your models. Consider prefixing with your organization or plugin name (e.g., `"myorg_lstm_v2"`).
*   **Implement the `BaseModel` Interface:** While the registry itself (currently) does not strictly enforce type checking at the point of registration (to avoid potential circular dependencies with `BaseModel` if it were to also know about the registry in detail), any model intended for use with `DigitalTwin` **must** conform to the `digital_twin_t1d.core.twin.BaseModel` interface. This ensures that `DigitalTwin` can interact with your model in a consistent way (e.g., call `fit()`, `predict()`, `get_params()`, `set_params()`). Ensure your custom model class inherits from `BaseModel` and correctly implements all its abstract methods and expected behaviors.
*   **Module Imports:** Remember that Python's import mechanism triggers the registration. If a module containing a `@register_model` decorator is never imported, the model within it will not be available.
*   **Logging:** The model registry uses Python's `logging` module to output information about registration, overwrites, and errors. Check your application logs if you encounter unexpected behavior.

By following these guidelines and utilizing the Model Registry, you can create robust and extensible predictive modeling solutions within the Digital Twin T1D framework.
```
