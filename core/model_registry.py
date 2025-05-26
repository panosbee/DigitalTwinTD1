"""
This module provides a registry for machine learning models.

It allows for registering, unregistering, and discovering models,
as well as retrieving model classes by name.

Enhanced for Digital Twin T1D SDK with diabetes-specific features.
"""

import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, Type, Optional, List
from dataclasses import dataclass

# Initialize logger
logger = logging.getLogger(__name__)

# Private global dictionary to store registered models
_MODEL_REGISTRY: Dict[str, Type[Any]] = {}


@dataclass
class ModelMetadata:
    """Metadata for registered models."""

    name: str
    model_class: Type[Any]
    version: str = "1.0.0"
    description: str = ""
    model_type: str = "glucose_predictor"  # glucose_predictor, meal_detector, exercise_impact, etc.
    clinical_validated: bool = False
    performance_metrics: Optional[Dict[str, float]] = None


# Enhanced registry with metadata
_MODEL_METADATA: Dict[str, ModelMetadata] = {}


class RegistrationError(Exception):
    """Custom exception for model registration errors."""

    pass


def register_model(
    model_name: str,
    overwrite: bool = False,
    version: str = "1.0.0",
    description: str = "",
    model_type: str = "glucose_predictor",
    clinical_validated: bool = False,
    performance_metrics: Optional[Dict[str, float]] = None,
) -> Callable[..., Type[Any]]:
    """
    Enhanced decorator to register a model class in the central registry.

    Args:
        model_name: The name to register the model under. Must be a non-empty string.
        overwrite: If True, allows overwriting an existing model with the same name.
        version: Model version (e.g., "1.0.0", "2.1.0")
        description: Human-readable description of the model
        model_type: Type of model (glucose_predictor, meal_detector, exercise_impact, etc.)
        clinical_validated: Whether the model has been clinically validated
        performance_metrics: Dict with performance metrics (e.g., {"mape": 4.9, "rmse": 15.2})

    Returns:
        A decorator function that registers the class.

    Raises:
        RegistrationError: If `model_name` is invalid, the object being registered
                           is not a class, or if a model with the same name already
                           exists and `overwrite` is False.
    """
    if not isinstance(model_name, str) or not model_name:
        raise RegistrationError("Model name must be a non-empty string.")

    def decorator(cls: Type[Any]) -> Type[Any]:
        if not isinstance(cls, type):
            raise RegistrationError(f"Object being registered for '{model_name}' is not a class.")

        if model_name in _MODEL_REGISTRY and not overwrite:
            raise RegistrationError(
                f"Model '{model_name}' is already registered. Use overwrite=True to replace."
            )

        # Register in both registries
        _MODEL_REGISTRY[model_name] = cls
        _MODEL_METADATA[model_name] = ModelMetadata(
            name=model_name,
            model_class=cls,
            version=version,
            description=description,
            model_type=model_type,
            clinical_validated=clinical_validated,
            performance_metrics=performance_metrics or {},
        )

        if overwrite and model_name in _MODEL_REGISTRY:
            logger.info("Model '%s' v%s overwritten.", model_name, version)
        else:
            logger.info("Model '%s' v%s registered (%s).", model_name, version, model_type)
        return cls

    return decorator


def get_model_class(model_name: str) -> Type[Any]:
    """
    Retrieves a model class from the registry by its name.

    Args:
        model_name: The name of the model to retrieve.

    Returns:
        The model class.

    Raises:
        RegistrationError: If the model is not found in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        available_models = list(_MODEL_REGISTRY.keys())
        raise RegistrationError(
            f"Model '{model_name}' not found in registry. " f"Available models: {available_models}"
        )
    return _MODEL_REGISTRY[model_name]


def get_model_metadata(model_name: str) -> ModelMetadata:
    """
    Retrieves model metadata from the registry.

    Args:
        model_name: The name of the model.

    Returns:
        ModelMetadata object with detailed information.

    Raises:
        RegistrationError: If the model is not found.
    """
    if model_name not in _MODEL_METADATA:
        raise RegistrationError(f"Model '{model_name}' not found in registry.")
    return _MODEL_METADATA[model_name]


def unregister_model(model_name: str) -> bool:
    """
    Unregisters a model from the registry.

    Args:
        model_name: The name of the model to unregister.

    Returns:
        True if the model was successfully unregistered, False otherwise.
    """
    success = False
    if model_name in _MODEL_REGISTRY:
        del _MODEL_REGISTRY[model_name]
        success = True
    if model_name in _MODEL_METADATA:
        del _MODEL_METADATA[model_name]
        success = True

    if success:
        logger.info("Model '%s' unregistered.", model_name)
    else:
        logger.warning("Model '%s' not found in registry for unregistration.", model_name)
    return success


def list_available_models() -> Dict[str, str]:
    """
    Lists all registered models and their class names.

    Returns:
        A dictionary where keys are model names and values are
        fully qualified class names (e.g., 'module.ClassName').
    """
    return {name: f"{cls.__module__}.{cls.__name__}" for name, cls in _MODEL_REGISTRY.items()}


def list_models_by_type(model_type: str) -> List[str]:
    """
    Lists models filtered by type.

    Args:
        model_type: The type of models to list (e.g., 'glucose_predictor', 'meal_detector')

    Returns:
        List of model names of the specified type.
    """
    return [name for name, metadata in _MODEL_METADATA.items() if metadata.model_type == model_type]


def get_best_model(model_type: str = "glucose_predictor", metric: str = "mape") -> Optional[str]:
    """
    Returns the best performing model of a given type based on a metric.

    Args:
        model_type: Type of model to search
        metric: Performance metric to optimize (lower is better for most metrics)

    Returns:
        Name of the best model, or None if no models found
    """
    candidates = []
    for name, metadata in _MODEL_METADATA.items():
        if (
            metadata.model_type == model_type
            and metadata.performance_metrics
            and metric in metadata.performance_metrics
        ):
            candidates.append((name, metadata.performance_metrics[metric]))

    if not candidates:
        return None

    # Sort by metric (ascending - lower is better)
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def get_clinical_validated_models() -> List[str]:
    """
    Returns list of clinically validated models.

    Returns:
        List of model names that have been clinically validated.
    """
    return [name for name, metadata in _MODEL_METADATA.items() if metadata.clinical_validated]


def is_model_registered(model_name: str) -> bool:
    """
    Checks if a model with the given name is registered.

    Args:
        model_name: The name of the model to check.

    Returns:
        True if the model is registered, False otherwise.
    """
    return model_name in _MODEL_REGISTRY


def discover_models(package_path: str = "models") -> None:
    """
    Discovers and registers models by importing modules in a specified package.

    This function walks through all modules in the given package_path,
    imports them, and relies on the @register_model decorator within those
    modules to add models to the registry.

    Args:
        package_path: The Python dot-path to the package containing models
                      (e.g., 'project.models', defaults to 'models').
    """
    try:
        package = importlib.import_module(package_path)
        modules_imported = 0

        for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if not is_pkg:  # Ensure it's a module, not a sub-package
                try:
                    importlib.import_module(name)
                    modules_imported += 1
                    logger.info("Successfully imported module: %s", name)
                except ImportError as e:
                    logger.error("Failed to import module %s: %s", name, e)

        models_count = len(_MODEL_REGISTRY)
        logger.info(
            "Model discovery completed. Imported %d modules, found %d registered models.",
            modules_imported,
            models_count,
        )

    except ImportError:
        logger.warning(
            "Package '%s' not found for model discovery. "
            "Ensure it's a valid Python package path and exists in PYTHONPATH.",
            package_path,
        )


def print_model_registry_summary() -> None:
    """
    Prints a summary of all registered models with their metadata.
    Useful for debugging and overview.
    """
    if not _MODEL_METADATA:
        print("No models registered.")
        return

    print(f"\n🧠 Digital Twin T1D - Model Registry Summary")
    print(f"{'='*60}")
    print(f"Total models: {len(_MODEL_METADATA)}")

    # Group by type
    by_type = {}
    for metadata in _MODEL_METADATA.values():
        if metadata.model_type not in by_type:
            by_type[metadata.model_type] = []
        by_type[metadata.model_type].append(metadata)

    for model_type, models in by_type.items():
        print(f"\n📊 {model_type.upper()} ({len(models)} models):")
        for metadata in models:
            status = "✅ Validated" if metadata.clinical_validated else "⏳ Research"
            metrics_str = ""
            if metadata.performance_metrics:
                metrics_str = " | ".join(
                    [f"{k}: {v}" for k, v in metadata.performance_metrics.items()]
                )
                metrics_str = f" | {metrics_str}"
            print(f"  - {metadata.name} v{metadata.version} [{status}]{metrics_str}")
            if metadata.description:
                print(f"    {metadata.description}")

    print(f"\n{'='*60}")
