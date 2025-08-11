"""
Author       : LIN Guocheng
Date         : 2024-04-04 00:16:04
LastEditors  : LIN Guocheng
LastEditTime : 2025-05-09 10:17:49
FilePath     : /root/lgc/RouterRL/src/rl_probabilistic_routing/jgat_mappo/agent/actor_critic.py
Description  : This module demonstrates the realization of JointGAT,
               a novel actor-critic network structure in our paper.
"""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch_geometric.nn import GATConv


class ActorCritic(nn.Module):
    """The realization of JointGAT, a novel actor-critic network structure.

    Args:
        local_edge_num (int): Number of local edges.
        num_features (int): Number of input features per node.
        action_dim (int): Dimension of the action space.
        action_std_init (float): Initial standard deviation for actions.
        buffer_size (int): Buffer size for experience replay.
        total_node_num (int): Total number of nodes.
        device (torch.device): Device for tensor computations (CPU or GPU).
    """

    def __init__(
        self,
        local_edge_num: int,
        num_features: int,
        action_dim: int,
        action_std_init: float,
        buffer_size: int,
        total_node_num: int,
        device: torch.device,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init**2).to(device)
        self.device = device
        self.local_edge_num = local_edge_num
        self.num_features = num_features
        self.buffer_size = buffer_size
        self.total_node_num = total_node_num

        gat_hidden_dim = 64

        # Initialize GAT layers
        self.gat1 = GATConv(num_features, gat_hidden_dim, heads=4)
        self.gat2 = GATConv(gat_hidden_dim * 4, gat_hidden_dim, heads=1)
        self.gat3 = GATConv(gat_hidden_dim, gat_hidden_dim, heads=1)

        # Actor network
        self.actor_fc1 = nn.Linear(gat_hidden_dim * (local_edge_num + 1), 64)
        self.actor_fc2 = nn.Linear(64, 32)
        self.actor_fc3 = nn.Linear(32, action_dim)

        # Critic network
        self.critic_fc1 = nn.Linear(gat_hidden_dim * self.total_node_num, 64)
        self.critic_fc2 = nn.Linear(64, 32)
        self.critic_fc3 = nn.Linear(32, 1)

    def set_action_std(self, new_action_std: float) -> None:
        """Update action standard deviation."""
        self.action_var = torch.full((self.action_dim,), new_action_std**2).to(self.device)

    def forward(self) -> None:
        """No forward pass defined. The act and evaluate methods should be used."""
        raise NotImplementedError

    def act(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """Generate an action and its log probability from the current policy.

        Args:
            node_features (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge index representing graph connections.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Action and its log probability.
        """
        x = torch.relu(self.gat1(node_features, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        x = torch.relu(self.gat3(x, edge_index))
        x_gat = x.view(-1, self.gat3.out_channels * (self.local_edge_num + 1))

        x = torch.relu(self.actor_fc1(x_gat))
        x = torch.relu(self.actor_fc2(x))
        action_mean = torch.softmax(self.actor_fc3(x), dim=-1)

        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(
        self,
        global_node_features: torch.Tensor,
        global_edge_indexes: torch.Tensor,
        node_features: torch.Tensor,
        edge_indexes: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the current policy and calculate state values.

        Args:
            global_node_features (torch.Tensor): Global node features.
            global_edge_indexes (torch.Tensor): Global edge indexes.
            node_features (torch.Tensor): Local node features.
            edge_indexes (torch.Tensor): Local edge indexes.
            actions (torch.Tensor): Actions taken by the agent.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Log probabilities, state values,
                                                             and action entropy.
        """
        state_values = []
        for (
            global_node_feature,
            global_edge_index,
            node_feature,
            edge_index,
            action,
        ) in zip(global_node_features, global_edge_indexes, node_features, edge_indexes, actions):
            # Evaluate actions
            x = torch.relu(self.gat1(node_feature, edge_index))
            x = torch.relu(self.gat2(x, edge_index))
            x = torch.relu(self.gat3(x, edge_index))
            x_gat = x.view(-1, self.gat3.out_channels * (self.local_edge_num + 1))

            x = torch.relu(self.actor_fc1(x_gat))
            x = torch.relu(self.actor_fc2(x))
            action_mean = self.actor_fc3(x)
            action_mean = torch.nan_to_num(action_mean, nan=1e-6, posinf=1.0, neginf=-1.0)
            action_var = torch.clamp(self.action_var.expand_as(action_mean), min=1e-6, max=1.0)
            dist = MultivariateNormal(action_mean, torch.diag_embed(action_var).to(self.device))
            logprob = dist.log_prob(action.unsqueeze(0) if action.dim() == 0 else action)

            # Evaluate state values
            x = torch.relu(self.gat1(global_node_feature, global_edge_index))
            x = torch.relu(self.gat2(x, global_edge_index))
            x = torch.relu(self.gat3(x, global_edge_index))
            x_gat = x.view(-1, self.gat3.out_channels * self.total_node_num)

            x = torch.relu(self.critic_fc1(x_gat))
            x = torch.relu(self.critic_fc2(x))
            state_value = self.critic_fc3(x)
            state_values.append(state_value)

        state_values = torch.cat(state_values, dim=0)
        return logprob, state_values, dist.entropy()
