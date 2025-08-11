"""
Author       : LIN Guocheng
Date         : 2024-04-04 00:16:04
LastEditors  : LIN Guocheng
LastEditTime : 2024-10-10 17:17:39
FilePath     : /root/lgc/RouterRL/src/rl_probabilistic_routing/jgat_mappo/agent/ppo.py
Description  : Implementation of JGAT-PPO, which combines Clipped Proximal Policy Optimization 
               with JointGAT.
"""

import sys
import torch
from torch import nn
import numpy

sys.path.append("src")
from rl_probabilistic_routing.jgat_mappo.agent.actor_critic import ActorCritic
from rl_probabilistic_routing.jgat_mappo.agent.rollout_buffer import RolloutBuffer


class PPO:
    """Implementation of Clipped Proximal Policy Optimization."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        joint_lr: float,
        gamma: float,
        k_epochs: int,
        eps_clip: float,
        device: torch.device,
        buffer_size: int,
        total_node_num: int,
        action_std_init: float = 0.6,
    ) -> None:
        self.learn_step = 0
        self.action_std = action_std_init
        self.observation_dim = observation_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.action_dim = action_dim
        self.buffer = RolloutBuffer(buffer_size)
        self.device = device

        self.policy = ActorCritic(
            observation_dim,
            2,
            action_dim,
            self.action_std,
            buffer_size,
            total_node_num,
            self.device,
        ).to(device)

        self.joint_optimizer = torch.optim.Adam(self.policy.parameters(), joint_lr)
        self.policy_old = ActorCritic(
            observation_dim,
            2,
            action_dim,
            self.action_std,
            buffer_size,
            total_node_num,
            self.device,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def set_action_std(self, new_action_std: float) -> None:
        """Update the action standard deviation."""
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float) -> None:
        """Decay the action standard deviation by a given rate until a minimum value."""
        self.action_std = max(min_action_std, self.action_std - action_std_decay_rate)
        self.set_action_std(self.action_std)

    def select_action(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select an action based on the current policy.

        Args:
            node_features (torch.Tensor): Node features for GAT input.
            edge_index (torch.Tensor): Edge indices representing the graph.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Action and log probability.
        """
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(node_features, edge_index)
        return action, action_logprob

    def add_buffer_sap(
        self,
        step: int,
        global_node_feature: torch.Tensor,
        global_edge_index: torch.Tensor,
        node_feature: torch.Tensor,
        edge_index: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
    ) -> None:
        """Add state-action-probability (SAP) data to the buffer."""
        self.buffer.add_sap(
            step,
            global_node_feature,
            global_edge_index,
            node_feature,
            edge_index,
            action,
            logprob,
        )

    def add_buffer_r(self, step: int, reward: float, delay: float, loss_rate: float) -> None:
        """Add reward-related data to the buffer."""
        self.buffer.add_r(step, reward, delay, loss_rate)

    def check_buffer_index(self, index: int) -> bool:
        """Check if a buffer index has valid data."""
        return self.buffer.check_buffer_index(index)

    def update(self) -> numpy.ndarray:
        """Perform policy update using the buffer data and return the joint loss.

        Returns:
            numpy.ndarray: Joint loss from policy optimization.
        """
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.buffer.get_all_values("reward")):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Prepare tensors
        def stack_and_detach(buffer: RolloutBuffer, key):
            return torch.stack(buffer.get_all_values(key)).detach().to(self.device)

        old_node_features = stack_and_detach(self.buffer, "node_feature")
        old_edge_indexes = stack_and_detach(self.buffer, "edge_index")
        old_global_node_features = stack_and_detach(self.buffer, "global_node_feature")
        old_global_edge_indexes = stack_and_detach(self.buffer, "global_edge_index")
        old_actions = stack_and_detach(self.buffer, "action")
        old_logprobs = stack_and_detach(self.buffer, "logprob")

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_global_node_features,
                old_global_edge_indexes,
                old_node_features,
                old_edge_indexes,
                old_actions,
            )
            state_values = torch.squeeze(state_values)

            # Surrogate loss calculation
            advantages = rewards - state_values.detach()
            surr1 = torch.exp(logprobs - old_logprobs.detach()) * advantages
            surr2 = (
                torch.clamp(
                    torch.exp(logprobs - old_logprobs.detach()),
                    1 - self.eps_clip,
                    1 + self.eps_clip,
                )
                * advantages
            )

            # Joint loss calculation
            joint_loss = (
                -torch.min(surr1, surr2)
                + 0.1 * dist_entropy
                + 0.5 * self.mse_loss(state_values, rewards)
            )

            # Perform optimization
            self.joint_optimizer.zero_grad()
            joint_loss.mean().backward()
            self.joint_optimizer.step()

        # Update old policy to new policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

        return joint_loss.mean().detach().cpu().numpy()

    def clean_buffer(self) -> None:
        """Clear the experience replay buffer."""
        self.buffer.clear()
