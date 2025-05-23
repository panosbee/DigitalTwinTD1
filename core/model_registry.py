"""
This module provides a registry for machine learning models.

It allows for registering, unregistering, and discovering models,
as well as retrieving model classes by name.
"""

import importlib
import logging
import pkgutil
from typing import Any, Callable, Dict, Type

# Initialize logger
logger = logging.getLogger(__name__)

# Private global dictionary to store registered models
_MODEL_REGISTRY: Dict[str, Type[Any]] = {}

class RegistrationError(Exception):
    """Custom exception for model registration errors."""
    pass


def register_model(model_name: str, overwrite: bool = False) -> Callable[..., Type[Any]]:
    """
    Decorator to register a model class in the central registry.

    Args:
        model_name: The name to register the model under. Must be a non-empty string.
        overwrite: If True, allows overwriting an existing model with the same name.
                   Defaults to False.

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

        _MODEL_REGISTRY[model_name] = cls
        if overwrite and model_name in _MODEL_REGISTRY:
            logger.info("Model '%s' overwritten.", model_name)
        else:
            logger.info("Model '%s' registered.", model_name)
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
        raise RegistrationError(f"Model '{model_name}' not found in registry.")
    return _MODEL_REGISTRY[model_name]


def unregister_model(model_name: str) -> bool:
    """
    Unregisters a model from the registry.

    Args:
        model_name: The name of the model to unregister.

    Returns:
        True if the model was successfully unregistered, False otherwise.
    """
    if model_name in _MODEL_REGISTRY:
        del _MODEL_REGISTRY[model_name]
        logger.info("Model '%s' unregistered.", model_name)
        return True
    logger.warning("Model '%s' not found in registry for unregistration.", model_name)
    return False


def list_available_models() -> Dict[str, str]:
    """
    Lists all registered models and their class names.

    Returns:
        A dictionary where keys are model names and values are
        fully qualified class names (e.g., 'module.ClassName').
    """
    return {
        name: f"{cls.__module__}.{cls.__name__}"
        for name, cls in _MODEL_REGISTRY.items()
    }


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
        for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            if not is_pkg:  # Ensure it's a module, not a sub-package
                try:
                    importlib.import_module(name)
                    logger.info("Successfully imported module: %s", name)
                except ImportError as e:
                    logger.error("Failed to import module %s: %s", name, e)
    except ImportError:
        logger.warning(
            "Package '%s' not found for model discovery. "
            "Ensure it's a valid Python package path and exists in PYTHONPATH.",
            package_path
        )
