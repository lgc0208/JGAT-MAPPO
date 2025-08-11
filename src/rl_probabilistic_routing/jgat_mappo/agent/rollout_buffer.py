"""
Author       : LIN Guocheng
Date         : 2024-04-04 00:16:04
LastEditors  : LIN Guocheng
LastEditTime : 2024-10-10 17:18:22
FilePath     : /root/lgc/RouterRL/src/rl_probabilistic_routing/jgat_mappo/agent/rollout_buffer.py
Description  : Realize experimence replay buffer for the DRL agent.
"""

import torch


class RolloutBuffer:
    """Experience replay buffer for the DRL agent."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.steps = {}
        self.step_order = []

    def _check_and_pop(self) -> None:
        """Check if buffer exceeds size and remove oldest element if it does."""
        if len(self.step_order) > self.buffer_size:
            oldest_step = self.step_order.pop(0)
            del self.steps[oldest_step]

    def add_sap(
        self,
        step: int,
        global_node_feature: torch.tensor,
        global_edge_index: torch.tensor,
        node_feature: torch.tensor,
        edge_index: torch.tensor,
        action: torch.tensor,
        logprob: torch.tensor,
    ) -> None:
        """Add state, action and logprob information into buffer.

        Args:
            step (int): The step index.
            node_feature (torch.tensor): node features in state.
            edge_index (torch.tensor): edge index in state.
            action (torch.tensor): action made by the agent.
            logprob (torch.tensor): log probability of action distribution.
        """
        step_data = {
            "global_node_feature": global_node_feature,
            "global_edge_index": global_edge_index,
            "node_feature": node_feature,
            "edge_index": edge_index,
            "action": action,
            "logprob": logprob,
            "reward": None,
            "delay": None,
            "loss_rate": None,
        }
        self.steps[step] = step_data
        self.step_order.append(step)
        self._check_and_pop()

    def add_r(self, step: int, reward: float, delay: float, loss_rate: float) -> None:
        """Add reward information into buffer.

        Args:
            step (int): The step index.
            reward (float): reward value after executing action.
            delay (float): delay value in reward.
            loss_rate (float): loss rate value in reward.
        """
        if step in self.steps:
            self.steps[step]["reward"] = reward
            self.steps[step]["delay"] = delay
            self.steps[step]["loss_rate"] = loss_rate
        else:
            raise ValueError(f"No SAPV data found for step {step}")

    def clear(self) -> None:
        """Clear data in replay buffer."""
        self.steps.clear()
        self.step_order.clear()

    def check_buffer_index(self, index: int) -> bool:
        """Check buffer at index whether NULL or not.

        Args:
            index (int): Index to check in buffer.

        Returns:
            bool: If none, return False; else True.
        """
        try:
            step_data = self.steps[index]
            return all(value is not None for key, value in step_data.items() if key != "step")
        except IndexError:
            return False

    def get_step_data(self, step: int):
        """Get the data for a specific step.

        Args:
            step (int): The step index to retrieve data for.

        Returns:
            dict: The data for the specified step.
        """
        if step in self.steps:
            return self.steps[step]
        raise ValueError(f"No data found for step {step}")

    def get_all_values(self, key: str):
        """Get all values for a given key in the order they were added.

        Args:
            key (str): The key for which to retrieve values.

        Returns:
            list: A list of all values for the given key in the order they were added.
        """
        values = []
        for step in self.step_order:
            if key in self.steps[step] and self.steps[step][key] is not None:
                values.append(self.steps[step][key])
        return values
