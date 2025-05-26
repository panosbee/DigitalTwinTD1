"""
🔌 Plugin System για Digital Twin SDK
====================================

Extensible plugin architecture για custom models, devices, και features.
"""

import importlib
import inspect
from typing import Dict, List, Any, Callable, Type, Optional
from abc import ABC, abstractmethod
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Base interface για όλα τα plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @abstractmethod
    def initialize(self, sdk) -> None:
        """Initialize plugin με SDK instance."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class ModelPlugin(PluginInterface):
    """Base class για custom prediction models."""

    @abstractmethod
    def predict(self, glucose_history: List[float], horizon_minutes: int) -> float:
        """Make glucose prediction."""
        pass

    @abstractmethod
    def train(self, training_data: Any) -> None:
        """Train the model."""
        pass

    @property
    def model_type(self) -> str:
        """Type of model (e.g., 'lstm', 'transformer')."""
        return "custom"


class DevicePlugin(PluginInterface):
    """Base class για custom device integrations."""

    @property
    @abstractmethod
    def device_type(self) -> str:
        """Device type identifier."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Connect to device."""
        pass

    @abstractmethod
    def get_current_glucose(self) -> float:
        """Get current glucose reading."""
        pass

    @abstractmethod
    async def stream_data(self):
        """Stream real-time data."""
        pass


class VisualizationPlugin(PluginInterface):
    """Base class για custom visualizations."""

    @abstractmethod
    def create_chart(self, data: Any) -> Any:
        """Create visualization."""
        pass

    @property
    def chart_type(self) -> str:
        """Type of chart."""
        return "custom"


class PluginManager:
    """Manage και load plugins."""

    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(exist_ok=True)

        self.plugins: Dict[str, PluginInterface] = {}
        self.model_plugins: Dict[str, ModelPlugin] = {}
        self.device_plugins: Dict[str, DevicePlugin] = {}
        self.viz_plugins: Dict[str, VisualizationPlugin] = {}

        # Plugin registry
        self.registry_file = self.plugin_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load plugin registry."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"plugins": {}, "enabled": [], "disabled": []}

    def _save_registry(self):
        """Save plugin registry."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def discover_plugins(self) -> List[str]:
        """Discover available plugins."""
        discovered = []

        # Check plugin directory
        for file in self.plugin_dir.glob("*.py"):
            if file.stem.startswith("_"):
                continue

            try:
                # Import module
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find plugin classes
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, PluginInterface)
                        and obj is not PluginInterface
                    ):

                        plugin_id = f"{file.stem}.{name}"
                        discovered.append(plugin_id)

                        # Update registry
                        if plugin_id not in self.registry["plugins"]:
                            self.registry["plugins"][plugin_id] = {
                                "file": str(file),
                                "class": name,
                                "discovered": True,
                            }

            except Exception as e:
                logger.error(f"Error discovering plugins in {file}: {e}")

        self._save_registry()
        return discovered

    def load_plugin(self, plugin_id: str, sdk_instance=None) -> bool:
        """Load a specific plugin."""
        if plugin_id not in self.registry["plugins"]:
            logger.error(f"Plugin {plugin_id} not found in registry")
            return False

        plugin_info = self.registry["plugins"][plugin_id]

        try:
            # Import module
            spec = importlib.util.spec_from_file_location(
                plugin_id.split(".")[0], plugin_info["file"]
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get plugin class
            plugin_class = getattr(module, plugin_info["class"])

            # Instantiate plugin
            plugin = plugin_class()

            # Initialize with SDK if provided
            if sdk_instance:
                plugin.initialize(sdk_instance)

            # Store plugin
            self.plugins[plugin.name] = plugin

            # Categorize plugin
            if isinstance(plugin, ModelPlugin):
                self.model_plugins[plugin.name] = plugin
            elif isinstance(plugin, DevicePlugin):
                self.device_plugins[plugin.name] = plugin
            elif isinstance(plugin, VisualizationPlugin):
                self.viz_plugins[plugin.name] = plugin

            # Update registry
            if plugin_id not in self.registry["enabled"]:
                self.registry["enabled"].append(plugin_id)

            self._save_registry()
            logger.info(f"✅ Loaded plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading plugin {plugin_id}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        if plugin_name not in self.plugins:
            return False

        try:
            # Cleanup plugin
            self.plugins[plugin_name].cleanup()

            # Remove from categories
            self.model_plugins.pop(plugin_name, None)
            self.device_plugins.pop(plugin_name, None)
            self.viz_plugins.pop(plugin_name, None)

            # Remove from main dict
            del self.plugins[plugin_name]

            logger.info(f"✅ Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Error unloading plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a specific plugin."""
        return self.plugins.get(name)

    def list_plugins(self) -> Dict[str, List[str]]:
        """List all loaded plugins by category."""
        return {
            "models": list(self.model_plugins.keys()),
            "devices": list(self.device_plugins.keys()),
            "visualizations": list(self.viz_plugins.keys()),
            "other": [
                name
                for name in self.plugins.keys()
                if name not in self.model_plugins
                and name not in self.device_plugins
                and name not in self.viz_plugins
            ],
        }

    def create_plugin_template(self, plugin_type: str, name: str) -> str:
        """Create a plugin template file."""
        templates = {
            "model": '''"""
Custom Model Plugin: {name}
==========================
"""

from sdk.plugins import ModelPlugin
import numpy as np


class {class_name}(ModelPlugin):
    """Custom glucose prediction model."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Custom glucose prediction model"
    
    def initialize(self, sdk) -> None:
        """Initialize with SDK instance."""
        self.sdk = sdk
        self.model = None  # Initialize your model here
        
    def predict(self, glucose_history: list, horizon_minutes: int) -> float:
        """Make glucose prediction."""
        # Implement your prediction logic
        if not glucose_history:
            return 120.0
            
        # Example: simple average
        return np.mean(glucose_history[-12:])  # Last hour
    
    def train(self, training_data) -> None:
        """Train the model."""
        # Implement your training logic
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
''',
            "device": '''"""
Custom Device Plugin: {name}
===========================
"""

from sdk.plugins import DevicePlugin
import asyncio
import random


class {class_name}(DevicePlugin):
    """Custom device integration."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Custom device integration"
    
    @property
    def device_type(self) -> str:
        return "{device_type}"
    
    def initialize(self, sdk) -> None:
        """Initialize with SDK instance."""
        self.sdk = sdk
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to device."""
        # Implement connection logic
        self.connected = True
        return True
    
    def get_current_glucose(self) -> float:
        """Get current glucose reading."""
        # Implement reading logic
        return random.uniform(80, 140)
    
    async def stream_data(self):
        """Stream real-time data."""
        while self.connected:
            yield {{
                "glucose": self.get_current_glucose(),
                "timestamp": datetime.now()
            }}
            await asyncio.sleep(300)  # 5 minutes
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.connected = False
''',
            "visualization": '''"""
Custom Visualization Plugin: {name}
==================================
"""

from sdk.plugins import VisualizationPlugin
import plotly.graph_objs as go


class {class_name}(VisualizationPlugin):
    """Custom visualization."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Custom visualization plugin"
    
    def initialize(self, sdk) -> None:
        """Initialize with SDK instance."""
        self.sdk = sdk
        
    def create_chart(self, data) -> go.Figure:
        """Create visualization."""
        # Implement your visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.get('x', []),
            y=data.get('y', []),
            mode='lines+markers',
            name='Custom Chart'
        ))
        
        fig.update_layout(
            title="{name} Visualization",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        
        return fig
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
''',
        }

        if plugin_type not in templates:
            raise ValueError(f"Unknown plugin type: {plugin_type}")

        # Create class name
        class_name = "".join(word.capitalize() for word in name.split("_"))
        if not class_name.endswith("Plugin"):
            class_name += "Plugin"

        # Format template
        content = templates[plugin_type].format(
            name=name, class_name=class_name, device_type=name.lower().replace(" ", "_")
        )

        # Save file
        filename = self.plugin_dir / f"{name.lower().replace(' ', '_')}_plugin.py"
        with open(filename, "w") as f:
            f.write(content)

        logger.info(f"✅ Created plugin template: {filename}")
        return str(filename)


# Decorators για easy plugin creation
def model_plugin(name: str, version: str = "1.0.0"):
    """Decorator για model plugins."""

    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            self._plugin_name = name
            self._plugin_version = version
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls.name = property(lambda self: self._plugin_name)
        cls.version = property(lambda self: self._plugin_version)

        return cls

    return decorator


def device_plugin(name: str, device_type: str, version: str = "1.0.0"):
    """Decorator για device plugins."""

    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            self._plugin_name = name
            self._plugin_version = version
            self._device_type = device_type
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls.name = property(lambda self: self._plugin_name)
        cls.version = property(lambda self: self._plugin_version)
        cls.device_type = property(lambda self: self._device_type)

        return cls

    return decorator


# Example plugin για demonstration
@model_plugin("Enhanced LSTM", "1.0.0")
class EnhancedLSTMPlugin(ModelPlugin):
    """Enhanced LSTM model με attention mechanism."""

    @property
    def description(self) -> str:
        return "LSTM with attention for better long-term predictions"

    def initialize(self, sdk) -> None:
        self.sdk = sdk
        # Initialize model here
        logger.info("Enhanced LSTM plugin initialized")

    def predict(self, glucose_history: List[float], horizon_minutes: int) -> float:
        # Placeholder prediction
        import numpy as np

        if not glucose_history:
            return 120.0

        # Simple moving average as placeholder
        recent = glucose_history[-12:]  # Last hour
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        current = glucose_history[-1]

        # Project forward
        steps = horizon_minutes // 5
        prediction = current + (trend * steps)

        return max(40, min(400, prediction))

    def train(self, training_data) -> None:
        logger.info("Training Enhanced LSTM model...")
        # Training logic here

    def cleanup(self) -> None:
        logger.info("Enhanced LSTM plugin cleanup")


# CLI για plugin management
def plugin_cli():
    """Simple CLI για plugin management."""
    import argparse

    parser = argparse.ArgumentParser(description="Digital Twin SDK Plugin Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover plugins")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load a plugin")
    load_parser.add_argument("plugin_id", help="Plugin ID to load")

    # List command
    list_parser = subparsers.add_parser("list", help="List loaded plugins")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create plugin template")
    create_parser.add_argument("type", choices=["model", "device", "visualization"])
    create_parser.add_argument("name", help="Plugin name")

    args = parser.parse_args()

    # Initialize manager
    manager = PluginManager()

    if args.command == "discover":
        plugins = manager.discover_plugins()
        print(f"Discovered {len(plugins)} plugins:")
        for p in plugins:
            print(f"  - {p}")

    elif args.command == "load":
        if manager.load_plugin(args.plugin_id):
            print(f"✅ Loaded {args.plugin_id}")
        else:
            print(f"❌ Failed to load {args.plugin_id}")

    elif args.command == "list":
        plugins = manager.list_plugins()
        for category, names in plugins.items():
            if names:
                print(f"\n{category.capitalize()}:")
                for name in names:
                    plugin = manager.get_plugin(name)
                    print(f"  - {name} v{plugin.version}: {plugin.description}")

    elif args.command == "create":
        filename = manager.create_plugin_template(args.type, args.name)
        print(f"✅ Created template: {filename}")

    else:
        parser.print_help()


if __name__ == "__main__":
    plugin_cli()
