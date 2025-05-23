"""
Base Agent Framework για Digital Twin T1D
========================================

Παρέχει κοινό interface για όλους τους reinforcement learning agents
που χρησιμοποιούνται για glucose control και insulin optimization.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import torch
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration για RL agents."""
    
    # Agent parameters
    agent_name: str = "BaseAgent"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Safety parameters
    enable_safety_layer: bool = True
    max_insulin_per_hour: float = 10.0  # units
    min_glucose_threshold: float = 70.0  # mg/dL
    max_glucose_threshold: float = 300.0  # mg/dL
    
    # Training parameters
    total_timesteps: int = 1000000
    log_interval: int = 1000
    eval_freq: int = 10000
    save_freq: int = 50000
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    seed: int = 42
    
    # Patient-specific parameters
    patient_weight: float = 70.0  # kg
    insulin_sensitivity: float = 50.0  # mg/dL per unit
    carb_ratio: float = 15.0  # grams carbs per unit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }


class BaseAgent(ABC):
    """
    Βασική κλάση για όλους τους RL agents στη διαχείριση διαβήτη.
    
    Παρέχει κοινό API για:
    - Action selection (act)
    - Learning από experience (learn) 
    - Model persistence (save/load)
    - Safety monitoring
    - Performance tracking
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.agent_name
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Initialize tracking variables
        self.total_steps = 0
        self.episode_count = 0
        self.training_history = []
        self.safety_violations = []
        self.last_action = None
        self.last_observation = None
        
        # Performance metrics
        self.performance_metrics = {
            'mean_reward': 0.0,
            'time_in_range': 0.0,
            'hypoglycemia_events': 0,
            'hyperglycemia_events': 0,
            'insulin_efficiency': 0.0,
            'safety_score': 1.0
        }
        
        logger.info(f"Initialized {self.name} agent on device {self.device}")
    
    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action based on current observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action to take (e.g., insulin dose)
        """
        pass
    
    @abstractmethod
    def learn(self, total_timesteps: int, **kwargs) -> Dict[str, float]:
        """
        Train the agent για specified number of timesteps.
        
        Args:
            total_timesteps: Number of environment steps για training
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent model και configuration.
        
        Args:
            path: File path για saving
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent model από saved file.
        
        Args:
            path: File path για loading
        """
        pass
    
    def apply_safety_constraints(self, action: np.ndarray, 
                                observation: np.ndarray) -> np.ndarray:
        """
        Apply safety constraints to action.
        
        Args:
            action: Raw action από policy
            observation: Current observation
            
        Returns:
            Safety-constrained action
        """
        if not self.config.enable_safety_layer:
            return action
        
        # Extract glucose από observation (assuming first element)
        current_glucose = observation[0] if len(observation) > 0 else 120.0
        
        # Constraint 1: No insulin if glucose too low
        if current_glucose < self.config.min_glucose_threshold:
            action = np.clip(action, 0, 0)  # No insulin delivery
            self._log_safety_violation("hypoglycemia_prevention", current_glucose, action)
        
        # Constraint 2: Limit maximum insulin
        action = np.clip(action, 0, self.config.max_insulin_per_hour)
        
        # Constraint 3: Emergency protocols για extreme glucose
        if current_glucose > self.config.max_glucose_threshold:
            # Allow higher insulin για severe hyperglycemia
            max_emergency_insulin = self.config.max_insulin_per_hour * 1.5
            action = np.clip(action, 0, max_emergency_insulin)
            self._log_safety_violation("hyperglycemia_emergency", current_glucose, action)
        
        return action
    
    def _log_safety_violation(self, violation_type: str, glucose: float, action: np.ndarray):
        """Log safety violation για monitoring."""
        violation = {
            'timestamp': datetime.now().isoformat(),
            'type': violation_type,
            'glucose': glucose,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'agent': self.name
        }
        self.safety_violations.append(violation)
        logger.warning(f"Safety violation: {violation_type} at glucose {glucose:.1f}")
    
    def update_performance_metrics(self, 
                                 reward: float,
                                 glucose: float,
                                 action: np.ndarray):
        """Update performance tracking metrics."""
        
        # Update cumulative metrics
        self.performance_metrics['mean_reward'] = (
            self.performance_metrics['mean_reward'] * 0.99 + reward * 0.01
        )
        
        # Track time in range
        if 70 <= glucose <= 180:
            self.performance_metrics['time_in_range'] = (
                self.performance_metrics['time_in_range'] * 0.99 + 1.0 * 0.01
            )
        else:
            self.performance_metrics['time_in_range'] = (
                self.performance_metrics['time_in_range'] * 0.99 + 0.0 * 0.01
            )
        
        # Track glycemic events
        if glucose < 70:
            self.performance_metrics['hypoglycemia_events'] += 0.01
        elif glucose > 180:
            self.performance_metrics['hyperglycemia_events'] += 0.01
        
        # Insulin efficiency (glucose reduction per unit insulin)
        if self.last_observation is not None and self.last_action is not None:
            insulin_delivered = np.sum(self.last_action)
            if insulin_delivered > 0:
                glucose_change = self.last_observation[0] - glucose
                efficiency = glucose_change / insulin_delivered
                self.performance_metrics['insulin_efficiency'] = (
                    self.performance_metrics['insulin_efficiency'] * 0.99 + efficiency * 0.01
                )
        
        # Update safety score based on violations
        recent_violations = len([v for v in self.safety_violations[-100:]])
        self.performance_metrics['safety_score'] = max(0.0, 1.0 - recent_violations / 100.0)
        
        # Store current state
        self.last_observation = np.array([glucose])
        self.last_action = action.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'agent_name': self.name,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'metrics': self.performance_metrics.copy(),
            'safety_violations': len(self.safety_violations),
            'recent_violations': len([
                v for v in self.safety_violations 
                if (datetime.now() - datetime.fromisoformat(v['timestamp'])).days < 7
            ]),
            'device': str(self.device),
            'config': self.config.to_dict()
        }
    
    def reset_episode(self):
        """Reset agent state για new episode."""
        self.episode_count += 1
        self.last_action = None
        self.last_observation = None
        logger.debug(f"Episode {self.episode_count} started για {self.name}")
    
    def evaluate_policy(self, 
                       env, 
                       n_episodes: int = 10, 
                       deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate agent policy on environment.
        
        Args:
            env: Gym environment για evaluation
            n_episodes: Number of episodes για evaluation
            deterministic: Use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        glucose_values = []
        tir_scores = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_glucose = []
            
            done = False
            while not done:
                action = self.act(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track glucose if available
                if 'glucose' in info:
                    episode_glucose.append(info['glucose'])
                elif len(obs) > 0:
                    episode_glucose.append(obs[0] * 100)  # Assuming normalized glucose
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            glucose_values.extend(episode_glucose)
            
            # Calculate time in range για this episode
            if episode_glucose:
                tir = np.mean([(70 <= g <= 180) for g in episode_glucose])
                tir_scores.append(tir)
        
        # Calculate evaluation metrics
        eval_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'time_in_range': np.mean(tir_scores) if tir_scores else 0.0,
            'mean_glucose': np.mean(glucose_values) if glucose_values else 0.0,
            'glucose_cv': np.std(glucose_values) / np.mean(glucose_values) if glucose_values else 0.0,
            'hypoglycemia_rate': np.mean([g < 70 for g in glucose_values]) if glucose_values else 0.0,
            'hyperglycemia_rate': np.mean([g > 180 for g in glucose_values]) if glucose_values else 0.0
        }
        
        logger.info(f"Evaluation completed: TIR={eval_metrics['time_in_range']:.1%}, "
                   f"Mean Reward={eval_metrics['mean_reward']:.2f}")
        
        return eval_metrics
    
    def __str__(self) -> str:
        return f"{self.name}(steps={self.total_steps}, episodes={self.episode_count})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"device='{self.device}', steps={self.total_steps})")


class MultiAgentCoordinator:
    """
    Coordinator για managing multiple agents σε ensemble setting.
    
    Allows for:
    - Ensemble decision making
    - Agent switching based on context
    - Performance comparison
    - Safety oversight
    """
    
    def __init__(self, agents: List[BaseAgent], voting_strategy: str = "weighted"):
        self.agents = agents
        self.voting_strategy = voting_strategy
        self.agent_weights = np.ones(len(agents)) / len(agents)
        self.performance_history = []
        
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get ensemble action από multiple agents."""
        actions = []
        weights = []
        
        for i, agent in enumerate(self.agents):
            try:
                action = agent.act(observation, deterministic)
                actions.append(action)
                weights.append(self.agent_weights[i])
            except Exception as e:
                logger.warning(f"Agent {agent.name} failed to act: {e}")
                continue
        
        if not actions:
            # Fallback to safe action
            return np.array([0.0])  # No insulin
        
        actions = np.array(actions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        if self.voting_strategy == "weighted":
            # Weighted average
            ensemble_action = np.average(actions, axis=0, weights=weights)
        elif self.voting_strategy == "majority":
            # Majority vote (για discrete actions)
            ensemble_action = np.median(actions, axis=0)
        elif self.voting_strategy == "best_performer":
            # Use action από best performing agent
            best_agent_idx = np.argmax(self.agent_weights)
            ensemble_action = actions[best_agent_idx]
        else:
            # Simple average
            ensemble_action = np.mean(actions, axis=0)
        
        return ensemble_action
    
    def update_agent_weights(self, rewards: List[float]):
        """Update agent weights based on recent performance."""
        if len(rewards) == len(self.agents):
            # Softmax weighting based on rewards
            exp_rewards = np.exp(np.array(rewards) - np.max(rewards))
            self.agent_weights = exp_rewards / exp_rewards.sum()
    
    def get_best_agent(self) -> BaseAgent:
        """Get currently best performing agent."""
        best_idx = np.argmax(self.agent_weights)
        return self.agents[best_idx]
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble performance."""
        agent_summaries = [agent.get_performance_summary() for agent in self.agents]
        
        return {
            'num_agents': len(self.agents),
            'voting_strategy': self.voting_strategy,
            'agent_weights': self.agent_weights.tolist(),
            'best_agent': self.get_best_agent().name,
            'agent_summaries': agent_summaries
        } 