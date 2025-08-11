"""
Author       : LIN Guocheng
Date         : 2024-04-05 19:46:47
LastEditors  : LIN Guocheng
LastEditTime : 2025-08-11 23:40:25
FilePath     : /root/lgc/JGAT-MAPPO/src/rl_probabilistic_routing/jgat_mappo/trainer.py
Description  : Training process implementation for JGAT-MAPPO in our network simulator.
"""

import json
import sys
from typing import List, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
from rich.console import Console
from rich.progress import track
from router_rl.probabilistic_env import ProbabilisticEnv

console = Console()
sys.path.append("src")
from rl_probabilistic_routing.jgat_mappo.agent.ppo import PPO


def get_local_edge_index_dim(node_index: int, edge_index: List[int]) -> int:
    """Get the dimension of the edge index for each agent.

    Args:
        node_index (int): Index of the current agent.
        edge_index (list[int]): Edge index from the routing table.

    Returns:
        int: Dimension of the agent's edge index.
    """
    return sum(1 for i in range(len(edge_index[0])) if edge_index[0, i].item() == node_index)


def get_local_subgraph(
    node_index: int, node_features: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the subgraph for each agent.

    Args:
        node_index (int): Index of the current agent.
        node_features (torch.Tensor): Node features for the entire graph.
        edge_index (torch.Tensor): Edge indices for the entire graph.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Local node features and edge indices.
    """
    neighbors = {
        edge_index[1, i].item()
        for i in range(len(edge_index[0]))
        if edge_index[0, i].item() == node_index
    }
    neighbors.add(node_index)

    neighbors = list(neighbors)
    node_map = {node: idx for idx, node in enumerate(neighbors)}
    local_node_features = node_features[neighbors, :]

    local_edge_index = [
        [node_map[edge_index[0, i].item()], node_map[edge_index[1, i].item()]]
        for i in range(edge_index.shape[1])
        if edge_index[0, i].item() in node_map and edge_index[1, i].item() in node_map
    ]
    local_edge_index = torch.tensor(
        local_edge_index, dtype=torch.long, device=node_features.device
    ).t()

    return local_node_features, local_edge_index


def worker(agent: PPO, res_queue: mp.Queue) -> None:
    """Multiprocessing worker to update PPO agents.

    Args:
        agent (PPO): PPO agent for the training.
        res_queue (mp.Queue): Queue to store joint losses.
    """
    res = agent.update()
    torch.cuda.empty_cache()
    res_queue.put(res)


def trainer(
    env: ProbabilisticEnv,
    node_num: int,
    action_dim_lst: List[int],
    max_ep_steps: int,
    max_episodes: int,
    joint_lr: float,
    gamma: float,
    k_epochs: int,
    eps_clip: float,
    summary_path: str,
    device: torch.device,
    action_std: float,
    action_std_decay_rate: float,
    min_action_std: float,
    utility_val: float,
    delay_mul: float,
    loss_rate_mul: float,
    cold_start_step: int = 5,
    reserve_cuda_memory_in_gb: float = 0,
    num_workers: int = 1,
) -> None:
    """Training process for JGAT-PPO in the proposed simulation framework.

    Args:
        env (ProbabilisticEnv): Environment object for the simulation framework.
        node_num (int): Number of nodes in the network topology.
        action_dim_lst (list[int]): Action dimensions for each node.
        max_ep_steps (int): Maximum number of steps per episode.
        max_episodes (int): Maximum number of episodes for training.
        joint_lr (float): Learning rate for JointGAT.
        gamma (float): Discount factor for PPO.
        k_epochs (int): Number of PPO update iterations.
        eps_clip (float): Clipping value for PPO.
        summary_path (str): Path for saving tensorboard summaries.
        device (torch.device): Device to use for training.
        action_std (float): Initial standard deviation for action distribution.
        action_std_decay_rate (float): Decay rate for action standard deviation.
        min_action_std (float): Minimum standard deviation for actions.
        utility_val (float): Utility function value for reward calculation.
        delay_mul (float): Weight for delay in reward function.
        loss_rate_mul (float): Weight for loss rate in reward function.
        cold_start_step (int): Number of steps before rewards are added to buffer.
        reserve_cuda_memory_in_gb (float): Amount of CUDA memory to reserve in GB.
        num_workers (int): Number of workers for parallel processing.
    """
    adjacent_matrix = torch.FloatTensor(env.routing_table).reshape(node_num, node_num).to(device)
    edge_indexes = (adjacent_matrix != 0).nonzero(as_tuple=False).t().contiguous()

    agents = []
    mp.set_start_method("spawn", force=True)

    for i in range(node_num):
        local_edge_index_dim = get_local_edge_index_dim(i, edge_indexes)
        agent = PPO(
            local_edge_index_dim,
            action_dim_lst[i],
            joint_lr,
            gamma,
            k_epochs,
            eps_clip,
            device,
            buffer_size=max_ep_steps,
            total_node_num=node_num,
            action_std_init=action_std,
        )
        agents.append(agent)

    summary_writer = SummaryWriter(summary_path)

    for current_episode in range(max_episodes):
        console.rule(f"Episode: {current_episode}")
        env.reset()
        ep_reward, ep_delay, ep_loss_rate, ep_mlu = 0, 0, 0, 0
        current_step = 0
        global_net_matrix = {}
        agent_rewards = [0 for _ in range(node_num)]

        while current_step < max_ep_steps:
            s_or_r, step, msg = env.get_obs()
            if s_or_r == "s":
                s = torch.FloatTensor(msg).to(device)
                load_matrix = s.reshape(node_num, node_num)
                in_load_avg = load_matrix.mean(axis=0)
                out_load_avg = load_matrix.mean(axis=1)
                node_features = torch.stack((in_load_avg, out_load_avg), dim=1)
                action_lst = []
                for index, agent in enumerate(agents):
                    local_node_features, local_edge_indexes = get_local_subgraph(
                        index, node_features, edge_indexes
                    )
                    a, p = agent.select_action(local_node_features, local_edge_indexes)
                    if step >= cold_start_step:
                        ep_mlu = max(s).cpu().item() if ep_mlu < max(s) else ep_mlu
                        agent.add_buffer_sap(
                            step - cold_start_step,
                            node_features,
                            edge_indexes,
                            local_node_features,
                            local_edge_indexes,
                            a,
                            p,
                        )
                    action_lst += a[0]

                action_str = ",".join(
                    [str(torch.sigmoid(action).cpu().item()) for action in action_lst]
                )
                env.make_action(action_str)
            else:
                env.reward_rcvd()
                if step >= cold_start_step:
                    reward_msgs = msg.split("/")
                    global_delay, global_loss_rate = map(float, reward_msgs[-1].split(","))
                    global_delay_utility = pow(global_delay, 1 - utility_val) / (1 - utility_val)
                    global_r = -delay_mul * global_delay_utility - loss_rate_mul * global_loss_rate
                    global_net_matrix[step - cold_start_step] = (
                        global_delay,
                        global_loss_rate,
                        global_r,
                    )
                    for index, agent in enumerate(agents):
                        delay, loss_rate = map(float, reward_msgs[index].split(","))
                        delay_utility = pow(delay, 1 - utility_val) / (1 - utility_val)
                        local_r = -delay_mul * delay_utility - loss_rate_mul * loss_rate
                        mix_r = 0.4 * local_r + 0.6 * global_r
                        agent.add_buffer_r(step - cold_start_step, mix_r, delay, loss_rate)

                    if all([agent.check_buffer_index(current_step) for agent in agents]):
                        delay, loss_rate, _ = global_net_matrix[current_step]
                        average_reward = (
                            sum(
                                agent.buffer.get_step_data(current_step)["reward"]
                                for agent in agents
                            )
                            / node_num
                        )

                        info = {
                            "Step": current_step,
                            "Delay": round(delay, 6),
                            "Loss Rate": round(loss_rate, 6),
                        }
                        console.log(json.dumps(info))

                        ep_reward += average_reward
                        ep_delay += delay
                        ep_loss_rate += loss_rate
                        current_step += 1

        ep_delay /= max_ep_steps
        ep_loss_rate /= max_ep_steps
        ep_reward /= max_ep_steps
        summary_writer.add_scalar("Average Reward", ep_reward, current_episode)
        summary_writer.add_scalar("Average Delay", ep_delay, current_episode)
        summary_writer.add_scalar("Average Loss Rate", ep_loss_rate, current_episode)
        summary_writer.add_scalar("Maximum Link Utilization", ep_mlu, current_episode)

        average_joint_loss = 0
        results_queue = mp.Queue()

        for i in range(0, len(agents), num_workers):
            processes = [
                mp.Process(target=worker, args=(agent, results_queue))
                for agent in agents[i : i + num_workers]
            ]
            for p in processes:
                p.start()

            for p in processes:
                p.join()

        joint_losses = []
        while not results_queue.empty():
            joint_losses.append(results_queue.get())

        for joint_loss, agent in zip(joint_losses, agents):
            average_joint_loss += joint_loss
            agent.clean_buffer()
            agent.decay_action_std(action_std_decay_rate, min_action_std)

        average_joint_loss /= node_num
        console.log(f"Episode: {current_episode}\tReward: {ep_reward}")
        summary_writer.add_scalar("Average Joint Loss", average_joint_loss, current_episode)

    if env:
        env.close()
    console.rule("Done!")
