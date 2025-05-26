"""
Intelligent Agents Package για Digital Twin T1D Library
=====================================================

Αυτό το package περιλαμβάνει state-of-the-art reinforcement learning agents
για optimal glucose control και insulin dosing.

Υλοποιεί:
- Safe Reinforcement Learning controllers
- PPO, SAC, TD3 algorithms με safety constraints
- Safety layers με intelligent shields
- Real-time deployment capabilities
"""

# Always available base classes
from .base import BaseAgent, AgentConfig, MultiAgentCoordinator

# Try to import SB3 wrappers
try:
    from .sb3_wrappers import make_sb3, PPOAgent, SACAgent, TD3Agent

    SB3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SB3 wrappers not available: {e}")
    SB3_AVAILABLE = False
    make_sb3 = PPOAgent = SACAgent = TD3Agent = None

# Try to import optional components
try:
    from .dual_ppo import DualPPOAgent
except ImportError:
    DualPPOAgent = None

try:
    from .safe_layer import SafetyLayer, HypoglycemiaShield, HyperglycemiaShield
except ImportError:
    SafetyLayer = HypoglycemiaShield = HyperglycemiaShield = None

try:
    from .offline_rl import ConservativeQLearning, ImplicitQLearning
except ImportError:
    ConservativeQLearning = ImplicitQLearning = None

try:
    from .environments import GlucoseControlEnv, SimGlucoseWrapper
except ImportError:
    GlucoseControlEnv = SimGlucoseWrapper = None

# Build __all__ dynamically based on what's available
__all__ = ["BaseAgent", "AgentConfig", "MultiAgentCoordinator"]

if SB3_AVAILABLE:
    __all__.extend(["make_sb3", "PPOAgent", "SACAgent", "TD3Agent"])

if DualPPOAgent:
    __all__.append("DualPPOAgent")

if SafetyLayer:
    __all__.extend(["SafetyLayer", "HypoglycemiaShield", "HyperglycemiaShield"])

if ConservativeQLearning:
    __all__.extend(["ConservativeQLearning", "ImplicitQLearning"])

if GlucoseControlEnv:
    __all__.extend(["GlucoseControlEnv", "SimGlucoseWrapper"])

__version__ = "2.0.0"
__author__ = "Digital Twin T1D Consortium"
__description__ = "State-of-the-art RL agents for diabetes management"
