import numpy as np
import argparse

# ------------------------
# Auction environment
# ------------------------
class Auction:
    def __init__(self, n_agents, n_items):
        self.n_agents = n_agents
        self.n_items = n_items

    def second_price_auction(self, bids, values):
        bids = np.array(bids)
        values = np.array(values)
        rewards = np.zeros_like(values)

        indices = np.arange(len(bids))
        np.random.shuffle(indices)
        sorted_idx = indices[np.argsort(bids[indices])[::-1]]

        price = bids[sorted_idx[self.n_items]]
        for i in range(self.n_items):
            winner = sorted_idx[i]
            rewards[winner] = values[winner] - price
        return rewards
        
def select_mode(mode, **overrides):
    config = {}
    if mode == "simple" or mode == "1":
        config = {
            "n_agents": 2,
            "n_items": 1,
            "n_episodes": 20000,
            "record_freq": 20000 / 500,
            "truth_max": 0.05,
            "num_points": 100
        }
    elif mode == "multi-agent" or mode == "2":
        config = {
            "n_agents": 4,
            "n_items": 1,
            "n_episodes": 30000,
            "record_freq": 30000 / 500,
            "truth_max": 0.10,
            "num_points": 100
        }
    elif mode == "multi-k-item" or mode == "3":
        config = {
            "n_agents": 5,
            "n_items": 3,
            "n_episodes": 30000,
            "record_freq": 25000 / 500,
            "truth_max": 0.10,
            "num_points": 50
        }
    elif mode == "multi-k-adversial" or mode == "4":
        config = {
            "n_agents": 5, # including adversial agents
            "n_items": 2,
            "n_episodes": 40000,
            "record_freq": 40000 / 500,
            "truth_max": 0.15,
            "num_points": 30
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
    parser.add_argument("--truth_max", type=float, help="Override truth max")
    parser.add_argument("--num_points", type=int, help="Override number of points")

    return parser.parse_args()


    