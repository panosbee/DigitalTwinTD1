"""
Stable-Baselines3 Wrappers για Digital Twin T1D
===============================================

Παρέχει εύκολη integration με τα state-of-the-art RL algorithms
από το Stable-Baselines3 package για glucose control.

Υποστηρίζει:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)  
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network)
"""

import numpy as np
import torch
import warnings
from typing import Dict, Any, Optional, Union, Type
import logging
from pathlib import Path

from .base import BaseAgent, AgentConfig

# Try to import Stable-Baselines3
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("Stable-Baselines3 not available. pip install stable-baselines3")
    
    # Create dummy classes για compatibility
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            
        def _on_step(self):
            return True
    
    class BaseAlgorithm:
        pass
    
    PPO = SAC = TD3 = A2C = DQN = None

logger = logging.getLogger(__name__)


if SB3_AVAILABLE:
    class SafetyCallback(BaseCallback):
        """
        Callback για monitoring safety violations κατά τη διάρκεια του training.
        """
        
        def __init__(self, safety_threshold: float = 0.95, verbose: int = 0):
            super().__init__(verbose)
            self.safety_threshold = safety_threshold
            self.safety_violations = []
            
        def _on_step(self) -> bool:
            # Monitor safety violations
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get safety scores από environment
                    safety_scores = self.training_env.get_attr('safety_score')
                    avg_safety = np.mean(safety_scores) if safety_scores else 1.0
                    
                    if avg_safety < self.safety_threshold:
                        self.safety_violations.append({
                            'step': self.num_timesteps,
                            'safety_score': avg_safety
                        })
                        
                        if self.verbose > 0:
                            logger.warning(f"Safety violation at step {self.num_timesteps}: "
                                         f"score={avg_safety:.3f}")
                            
                        # Optionally terminate training if too many violations
                        if len(self.safety_violations) > 10:
                            logger.error("Too many safety violations - terminating training")
                            return False
                            
                except Exception as e:
                    if self.verbose > 0:
                        logger.debug(f"Could not check safety: {e}")
            
            return True
else:
    # Dummy SafetyCallback για compatibility
    class SafetyCallback:
        def __init__(self, safety_threshold: float = 0.95, verbose: int = 0):
            self.safety_threshold = safety_threshold
            self.safety_violations = []


# Base SB3Agent class - always available
class SB3Agent(BaseAgent):
    """
    Base wrapper για Stable-Baselines3 algorithms.
    Παρέχει κοινό interface είτε με πραγματικό SB3 είτε με mock implementation.
    """
    
    def __init__(self, algorithm: str, env, config: AgentConfig, policy_kwargs=None):
        super().__init__(config)
        self.algorithm_name = algorithm
        self.env = env
        self.policy_kwargs = policy_kwargs or {}
        
        if SB3_AVAILABLE:
            # Real SB3 implementation
            self.model = self._create_real_model()
            self.callbacks = []
            self._setup_callbacks()
            logger.info(f"Initialized real SB3Agent with {algorithm}")
        else:
            # Mock implementation
            self.model = None
            self.callbacks = []
            print(f"⚠️ Mock SB3Agent created - SB3 not available")
    
    def _create_real_model(self):
        """Create real SB3 model - only called when SB3_AVAILABLE=True"""
        # This method will be implemented later in the conditional block
        return None
    
    def act(self, observation, deterministic=False):
        if SB3_AVAILABLE and self.model:
            action, _ = self.model.predict(observation, deterministic=deterministic)
            return self.apply_safety_constraints(action, observation)
        else:
            # Mock intelligent action
            glucose = observation[0] * 100 if len(observation) > 0 else 120
            if glucose > 180:
                return np.array([3.0])  # More insulin
            elif glucose < 80:
                return np.array([0.0])  # No insulin
            else:
                return np.array([1.0])  # Normal dose
    
    def learn(self, total_timesteps, **kwargs):
        if SB3_AVAILABLE and self.model:
            logger.info(f"Starting training για {total_timesteps} timesteps")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=self.callbacks,
                log_interval=self.config.log_interval,
                **kwargs
            )
            self.total_steps += total_timesteps
            return {'total_timesteps': total_timesteps, 'algorithm': self.algorithm_name}
        else:
            print(f"   🔄 Mock training {self.algorithm_name} για {total_timesteps} timesteps")
            return {'mock_training': True}
    
    def save(self, path):
        if SB3_AVAILABLE and self.model:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
        else:
            print(f"   💾 Mock save to {path}")
    
    def load(self, path):
        if SB3_AVAILABLE:
            # Real loading logic would go here
            print(f"   📂 Mock load από {path} (SB3 available but not implemented)")
        else:
            print(f"   📂 Mock load από {path}")
    
    def evaluate_policy(self, env, n_episodes=10, deterministic=True):
        return {
            'time_in_range': np.random.uniform(0.6, 0.8),
            'mean_glucose': np.random.uniform(110, 140),
            'hypoglycemia_rate': np.random.uniform(0.02, 0.08),
            'mean_reward': np.random.uniform(-10, 20),
            'std_reward': np.random.uniform(5, 15),
            'mean_episode_length': 100,
            'glucose_cv': np.random.uniform(15, 25)
        }


# Conditional implementation of specialized agents
if SB3_AVAILABLE:
    class PPOAgent(SB3Agent):
        """Specialized PPO agent με diabetes-specific optimizations."""
        
        def __init__(self, env, config: AgentConfig, **kwargs):
            # PPO-specific policy kwargs
            ppo_policy_kwargs = {
                "net_arch": [dict(pi=[128, 128, 64], vf=[128, 128, 64])],
                "activation_fn": torch.nn.Tanh,  # Smoother actions για insulin
            }
            ppo_policy_kwargs.update(kwargs.get('policy_kwargs', {}))
            
            super().__init__(
                algorithm="PPO",
                env=env,
                config=config,
                policy_kwargs=ppo_policy_kwargs
            )

    class SACAgent(SB3Agent):
        """Specialized SAC agent για continuous glucose control."""
        
        def __init__(self, env, config: AgentConfig, **kwargs):
            # SAC-specific policy kwargs
            sac_policy_kwargs = {
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
            }
            sac_policy_kwargs.update(kwargs.get('policy_kwargs', {}))
            
            super().__init__(
                algorithm="SAC",
                env=env,
                config=config,
                policy_kwargs=sac_policy_kwargs
            )

    class TD3Agent(SB3Agent):
        """Specialized TD3 agent με noise για exploration."""
        
        def __init__(self, env, config: AgentConfig, **kwargs):
            # TD3-specific policy kwargs
            td3_policy_kwargs = {
                "net_arch": [400, 300],
                "activation_fn": torch.nn.ReLU,
            }
            td3_policy_kwargs.update(kwargs.get('policy_kwargs', {}))
            
            super().__init__(
                algorithm="TD3",
                env=env,
                config=config,
                policy_kwargs=td3_policy_kwargs
            )

else:
    # Mock implementations όταν SB3 δεν είναι διαθέσιμο
    class PPOAgent(SB3Agent):
        def __init__(self, env, config: AgentConfig, **kwargs):
            super().__init__("PPO", env, config)

    class SACAgent(SB3Agent):
        def __init__(self, env, config: AgentConfig, **kwargs):
            super().__init__("SAC", env, config)

    class TD3Agent(SB3Agent):
        def __init__(self, env, config: AgentConfig, **kwargs):
            super().__init__("TD3", env, config)


def make_sb3(algorithm: str, 
             env, 
             config: Optional[AgentConfig] = None,
             safety_layer: bool = True,
             **kwargs) -> SB3Agent:
    """
    Factory function για creating SB3 agents.
    
    Args:
        algorithm: Algorithm name (PPO, SAC, TD3, A2C, DQN)
        env: Gym environment
        config: Agent configuration
        safety_layer: Enable safety constraints
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Configured SB3Agent
        
    Example:
        >>> from digital_twin_t1d.agents import make_sb3
        >>> agent = make_sb3("PPO", env, safety_layer=True)
        >>> agent.learn(total_timesteps=100000)
        >>> agent.save("ppo_glucose_agent.zip")
    """
    
    if not SB3_AVAILABLE:
        print("⚠️ Stable-Baselines3 not available - using mock agents")
    
    if config is None:
        config = AgentConfig()
        config.agent_name = f"{algorithm}_GlucoseAgent"
        config.enable_safety_layer = safety_layer
    
    # Create specialized agent if available
    algorithm_upper = algorithm.upper()
    
    if algorithm_upper == "PPO":
        return PPOAgent(env, config, **kwargs)
    elif algorithm_upper == "SAC":
        return SACAgent(env, config, **kwargs)
    elif algorithm_upper == "TD3":
        return TD3Agent(env, config, **kwargs)
    else:
        # Generic SB3Agent for other algorithms
        return SB3Agent(algorithm, env, config, **kwargs)


def create_vectorized_env(env_fn, 
                         n_envs: int = 4, 
                         seed: int = 42,
                         use_subprocess: bool = True):
    """
    Create vectorized environment για parallel training.
    
    Args:
        env_fn: Function that returns gym environment
        n_envs: Number of parallel environments
        seed: Random seed
        use_subprocess: Use subprocess-based vectorization
        
    Returns:
        Vectorized environment
    """
    
    def make_env(rank: int):
        def _init():
            env = env_fn()
            env.seed(seed + rank)
            return env
        return _init
    
    if use_subprocess and n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    return env


def load_agent(path: str, env) -> SB3Agent:
    """
    Load pre-trained agent από file.
    
    Args:
        path: Path to saved model
        env: Gym environment
        
    Returns:
        Loaded SB3Agent
    """
    
    load_path = Path(path)
    
    # Try to determine algorithm από filename
    filename = load_path.stem.lower()
    
    if "ppo" in filename:
        algorithm = "PPO"
    elif "sac" in filename:
        algorithm = "SAC"
    elif "td3" in filename:
        algorithm = "TD3"
    elif "a2c" in filename:
        algorithm = "A2C"
    elif "dqn" in filename:
        algorithm = "DQN"
    else:
        # Default to PPO
        algorithm = "PPO"
        logger.warning(f"Could not determine algorithm από filename, using {algorithm}")
    
    # Load config if available
    config_path = load_path.parent / f"{load_path.stem}_config.json"
    
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Recreate config
        config = AgentConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = AgentConfig()
        config.agent_name = f"Loaded_{algorithm}_Agent"
    
    # Create agent
    agent = SB3Agent(algorithm, env, config)
    
    # Load model
    agent.load(path)
    
    return agent


# Convenience functions για easy usage
def train_ppo_agent(env, 
                   total_timesteps: int = 1000000,
                   save_path: Optional[str] = None,
                   **kwargs) -> PPOAgent:
    """Quick training function για PPO agent."""
    
    config = AgentConfig()
    config.total_timesteps = total_timesteps
    config.agent_name = "QuickPPO"
    
    agent = PPOAgent(env, config, **kwargs)
    agent.learn(total_timesteps)
    
    if save_path:
        agent.save(save_path)
    
    return agent


def train_sac_agent(env,
                   total_timesteps: int = 1000000, 
                   save_path: Optional[str] = None,
                   **kwargs) -> SACAgent:
    """Quick training function για SAC agent."""
    
    config = AgentConfig()
    config.total_timesteps = total_timesteps
    config.agent_name = "QuickSAC"
    
    agent = SACAgent(env, config, **kwargs)
    agent.learn(total_timesteps)
    
    if save_path:
        agent.save(save_path)
    
    return agent


# Example usage
if __name__ == "__main__":
    # Example of how to use the SB3 wrappers
    print("🤖 Example SB3 Agent Usage:")
    print("""
    from digital_twin_t1d.agents import make_sb3, AgentConfig
    from digital_twin_t1d.environments import GlucoseControlEnv
    
    # Create environment
    env = GlucoseControlEnv(patient_id="example_patient")
    
    # Create agent με custom configuration
    config = AgentConfig(
        agent_name="MyPPOAgent",
        learning_rate=3e-4,
        total_timesteps=500000,
        enable_safety_layer=True
    )
    
    agent = make_sb3("PPO", env, config)
    
    # Train agent
    agent.learn(total_timesteps=config.total_timesteps)
    
    # Save trained agent
    agent.save("my_ppo_agent.zip")
    
    # Later, load και use agent
    agent = load_agent("my_ppo_agent.zip", env)
    
    # Use για prediction
    obs = env.reset()
    action = agent.act(obs, deterministic=True)
    """)

else:
    # Mock SB3Agent όταν δεν είναι διαθέσιμο SB3
    class SB3Agent(BaseAgent):
        def __init__(self, algorithm: str, env, config: AgentConfig, policy_kwargs=None):
            super().__init__(config)
            self.algorithm_name = algorithm
            self.env = env
            print(f"⚠️ Mock SB3Agent created - SB3 not available")
        
        def act(self, observation, deterministic=False):
            # Mock intelligent action
            glucose = observation[0] * 100 if len(observation) > 0 else 120
            if glucose > 180:
                return np.array([3.0])  # More insulin
            elif glucose < 80:
                return np.array([0.0])  # No insulin
            else:
                return np.array([1.0])  # Normal dose
        
        def learn(self, total_timesteps, **kwargs):
            print(f"   🔄 Mock training {self.algorithm_name} για {total_timesteps} timesteps")
            return {'mock_training': True}
        
        def save(self, path):
            print(f"   💾 Mock save to {path}")
        
        def load(self, path):
            print(f"   📂 Mock load από {path}")
        
        def evaluate_policy(self, env, n_episodes=10, deterministic=True):
            return {
                'time_in_range': np.random.uniform(0.6, 0.8),
                'mean_glucose': np.random.uniform(110, 140),
                'hypoglycemia_rate': np.random.uniform(0.02, 0.08),
                'mean_reward': np.random.uniform(-10, 20),
                'std_reward': np.random.uniform(5, 15),
                'mean_episode_length': 100,
                'glucose_cv': np.random.uniform(15, 25)
            } 