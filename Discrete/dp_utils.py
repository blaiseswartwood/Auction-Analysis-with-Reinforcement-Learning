import numpy as np
import argparse

def select_mode(mode, **overrides):
    config = {}
    if mode == "simple" or mode == "1":
        config = {
            "n_agents": 2,
            "n_items": 1,
            "n_episodes": 1000000,
            "n_state": 101,
            "n_action": 101
        }
    elif mode == "multi-agent" or mode == "2":
        config = {
            "n_agents": 4,
            "n_items": 1,
            "n_episodes": 1000000,
            "n_state": 20,
            "n_action": 20
        }
    elif mode == "multi-k-item" or mode == "3":
        config = {
            "n_agents": 5,
            "n_items": 2,
            "n_episodes": 1000000,
            "n_state": 11,
            "n_action": 11
        }
    elif mode == "multi-k-adversial" or mode == "4":
        config = {
            "n_agents": 5, # including adversial agents
            "n_items": 2,
            "n_episodes": 1000000,
            "n_state": 11,
            "n_action": 11
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
    config.update(overrides)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Auction Training Script")

    parser.add_argument(
        "--mode",
        type=str,
        default="1",
        help="Select a mode: 1 (simple), 2 (multi-agent), 3 (multi-k-item), 4 (multi-k-adversial)"
    )

    parser.add_argument("--n_agents", type=int, help="Override number of agents")
    parser.add_argument("--n_items", type=int, help="Override number of items")
    parser.add_argument("--n_episodes", type=int, help="Override number of episodes")
    parser.add_argument("--n_state", type=float, help="Override number of states")
    parser.add_argument("--n_action", type=int, help="Override number of actions")

    return parser.parse_args()


    