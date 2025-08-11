"""
Author       : LIN Guocheng
Date         : 2024-04-05 19:46:47
LastEditors  : LIN Guocheng
LastEditTime : 2025-08-11 23:39:43
FilePath     : /root/lgc/JGAT-MAPPO/src/rl_probabilistic_routing/jgat_mappo/main.py
Description  : Main program for JGAT-MAPPO, operating in the KDN-based Network Simulation Framework.
"""

import argparse
import itertools
import json
import os
import re
import sys
import time
from argparse import Namespace
from typing import List

import numpy as np
import torch
from torch import multiprocessing
from rich.console import Console
from rich.traceback import install

install(show_locals=True)
console = Console()

sys.path.append(".")
from router_rl.probabilistic_env import ProbabilisticEnv

sys.path.append("src")
from rl_probabilistic_routing.jgat_mappo.trainer import trainer


def get_action_dim(env: ProbabilisticEnv, agent_num: int) -> List[int]:
    """Get action dimensions for each agent based on the routing table.

    Args:
        env (ProbabilisticEnv): The network environment.
        agent_num (int): Number of agents (nodes) in the network.

    Returns:
        List[int]: Action dimensions for each agent.
    """
    action_dims = np.zeros(agent_num, dtype=int)
    routing_table = np.array(env.routing_table).reshape((agent_num, agent_num))

    for i, j in itertools.product(range(agent_num), range(agent_num)):
        if routing_table[i][j] != 0:
            action_dims[i] += 1

    return action_dims.tolist()


if __name__ == "__main__":
    # Load hyperparameters from JSON file
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/hyperparameters.json", "r", encoding="utf-8"
    ) as f:
        defaults = json.load(f)

    # Argument parser with default values from the JSON
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing_mode", type=str, default=defaults["routing_mode"])
    parser.add_argument("--return_mode", type=str, default=defaults["return_mode"])
    parser.add_argument("--docker_name", type=str, default=defaults["docker_name"])
    parser.add_argument("--algorithm", type=str, default=defaults["algorithm"])
    parser.add_argument("--tensorboard_label", type=str, default=defaults["tensorboard_label"])
    parser.add_argument("--max_episodes", type=int, default=defaults["max_episodes"])
    parser.add_argument("--max_ep_steps", type=int, default=defaults["max_ep_steps"])
    parser.add_argument("--joint_lr", type=float, default=defaults["joint_lr"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--k_epochs", type=int, default=defaults["k_epochs"])
    parser.add_argument("--eps_clip", type=float, default=defaults["eps_clip"])
    parser.add_argument("--action_std", type=float, default=defaults["action_std"])
    parser.add_argument(
        "--action_std_decay_rate", type=float, default=defaults["action_std_decay_rate"]
    )
    parser.add_argument("--min_action_std", type=float, default=defaults["min_action_std"])
    parser.add_argument("--utility_val", type=float, default=defaults["utility_val"])
    parser.add_argument("--delay_mul", type=int, default=defaults["delay_mul"])
    parser.add_argument("--loss_rate_mul", type=int, default=defaults["loss_rate_mul"])
    parser.add_argument("--cold_start_step", type=int, default=defaults["cold_start_step"])
    parser.add_argument(
        "--reserve_cuda_memory_in_gb", type=float, default=defaults["reserve_cuda_memory_in_gb"]
    )
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--topology", type=str, default=defaults["topology"])
    parser.add_argument("--flow_rate", type=float, default=defaults["flow_rate"])
    parser.add_argument("--ned_path", type=str, default=defaults["ned_path"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])

    args: Namespace = parser.parse_args()

    # Initialize basic parameters
    with open(f"{args.ned_path}/{args.topology}.ned", "r", encoding="utf-8") as file:
        ned_content = file.read()

    router_match = re.search(r"R\[(\d+)]", ned_content)
    node_num = int(router_match.group(1)) if router_match else 0
    edge_num = len(re.findall(r"R\[(\d+)].*? <--> C <--> R\[(\d+)].*?;", ned_content))

    local_time = time.localtime(time.mktime(time.gmtime()) + 8 * 3600)
    date = time.strftime("%Y%m%d", local_time)
    now_time = time.strftime("%H-%M-%S", local_time)

    # Set Tensorboard path
    algorithm = f"test/{args.algorithm}" if args.test else args.algorithm
    if args.tensorboard_label:
        algorithm = f"{algorithm}/{args.tensorboard_label}"

    seed_suffix = f"-seed{args.seed}" if args.seed != 0 else ""
    timestamp = f"{date}-{now_time}"
    summary_path = (
        f"/root/data/runs/{algorithm}-fr{args.flow_rate}-{args.topology}{seed_suffix}-{timestamp}"
    )
    network_log_path = (
        f"{os.getcwd()}/logs/inet/{algorithm}-fr{args.flow_rate}-{args.topology}-{timestamp}"
    )
    console.log(f"summary path: {summary_path}")
    console.log(f"network log path: {network_log_path}")

    # Set up GPU
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    multiprocessing.set_start_method("spawn")

    # Initialize environment
    env = ProbabilisticEnv(
        network=args.topology,
        flow_rate=args.flow_rate,
        total_step=args.max_ep_steps,
        routing_mode=args.routing_mode,
        return_mode=args.return_mode,
        seed=args.seed,
        ned_path=args.ned_path,
        log_path=network_log_path,
    )

    action_dim_lst = get_action_dim(env, node_num)

    # Start the trainer
    trainer(
        env,
        node_num,
        action_dim_lst,
        args.max_ep_steps,
        args.max_episodes,
        args.joint_lr,
        args.gamma,
        args.k_epochs,
        args.eps_clip,
        summary_path,
        device,
        args.action_std,
        args.action_std_decay_rate,
        args.min_action_std,
        args.utility_val,
        args.delay_mul,
        args.loss_rate_mul,
        args.cold_start_step,
        args.reserve_cuda_memory_in_gb,
        args.num_workers,
    )

    if env:
        env.close()
