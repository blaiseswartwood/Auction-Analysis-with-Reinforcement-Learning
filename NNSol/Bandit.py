import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------
# Q-network with LayerNorm and larger capacity
# ------------------------
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.fc2(x)))
        return self.fc3(x)

# ------------------------
# Improved actor network
# ------------------------
class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # bid in [0, 1]
        )

    def forward(self, state):
        return self.net(state)

# ------------------------
# Bandit Agent
# ------------------------
class BanditAgent:
    def __init__(self, state_dim=1, action_dim=1, lr=1e-3):
        self.qnet = QNet(state_dim, action_dim).to(device)
        self.actor = Actor(state_dim).to(device)
        self.qopt = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def update(self, state, action, reward):
        state = torch.FloatTensor(np.atleast_2d(state)).to(device)
        action = torch.FloatTensor(np.atleast_2d(action)).to(device)
        reward = torch.FloatTensor(np.atleast_2d(reward)).to(device)

        pred = self.qnet(state, action)
        loss = F.mse_loss(pred, reward)

        self.qopt.zero_grad()
        loss.backward()
        self.qopt.step()
        return loss.item()

    def update_actor(self, state):
        state = torch.FloatTensor(np.atleast_2d(state)).to(device)
        bid = self.actor(state)
        pred_reward = self.qnet(state, bid)

        # Optional entropy regularization to encourage exploration
        entropy = - (bid * torch.log(bid + 1e-8) + (1 - bid) * torch.log(1 - bid + 1e-8)).mean()
        loss = -pred_reward.mean() - 0.01 * entropy

        self.aopt.zero_grad()
        loss.backward()
        self.aopt.step()
        return loss.item()

    def select_action(self, state, n_samples=10):
        state_tensor = torch.FloatTensor(np.atleast_2d(state)).to(device)
        sampled_actions = []
        predicted_rewards = []

        self.qnet.train()  

        for _ in range(n_samples):
            bid = torch.rand((1, 1)).to(device)
            with torch.no_grad():
                q_val = self.qnet(state_tensor, bid)
            sampled_actions.append(bid.item())
            predicted_rewards.append(q_val.item())

        self.qnet.eval()  

        best_idx = np.argmax(predicted_rewards)
        return np.clip(sampled_actions[best_idx], 0.0, 1.0)

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
            "n_agents": 5,
            "n_items": 1,
            "n_episodes": 30000,
            "record_freq": 30000 / 500,
            "truth_max": 0.1,
            "num_points": 150
        }
    elif mode == "multi-k-item" or mode == "3":
        config = {
            "n_agents": 5,
            "n_items": 3,
            "n_episodes": 25000,
            "record_freq": 25000 / 500,
            "truth_max": 0.07,
            "num_points": 120
        }
    elif mode == "multi-k-adversial" or mode == "4":
        config = {
            "n_agents": 6,
            "n_items": 3,
            "n_episodes": 40000,
            "record_freq": 40000 / 500,
            "truth_max": 0.15,
            "num_points": 200
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
    
# ------------------------
# Main training loop
# ------------------------
if __name__ == "__main__":
    args = parse_args()
    overrides = {
        k: v for k, v in vars(args).items()
        if k != "mode" and v is not None
    }
    config = select_mode(args.mode, **overrides)

    n_agents = config["n_agents"]
    n_items = config["n_items"]
    n_episodes = config["n_episodes"]
    record_freq = config["record_freq"]
    truth_max = config["truth_max"]
    num_points = config["num_points"]

    print(f"Mode: {args.mode}")
    print(f"Agents: {n_agents}, Items: {n_items}, Episodes: {n_episodes}, Truth max: {truth_max}, Points: {num_points}")
    
    agents = [BanditAgent() for _ in range(n_agents)]
    auction = Auction(n_agents, n_items)

    reward_list = [[] for _ in range(n_agents)]
    loss_q_list = [[] for _ in range(n_agents)]
    loss_act_list = [[] for _ in range(n_agents)]

    for ep in tqdm(range(n_episodes)):
        values = [np.random.uniform(0, 1) for _ in range(n_agents)]
        bids = [agents[i].select_action([values[i]]) for i in range(n_agents)]
        rewards = auction.second_price_auction(bids, values)

        for i in range(n_agents):
            loss_q = agents[i].update([values[i]], [bids[i]], [rewards[i]])
            loss_act = agents[i].update_actor([values[i]])

            # Recording
            if ep % record_freq == 0:
                loss_q_list[i].append(loss_q)
                loss_act_list[i].append(loss_act)
                reward_list[i].append(rewards[i])

    # ------------------------
    # Evaluation
    # ------------------------
    for i, agent in enumerate(agents):
        print(f"\nAgent {i} bidding policy:")
        truthful = 0
        for v in np.linspace(0, 1, num_points):
            bid = agent.actor(torch.tensor([[v]], dtype=torch.float32).to(device)).item()
            print(f"Value: {v:.2f} → Bid: {bid:.2f}")
            if abs(bid - v) <= truth_max:
                truthful += 1
        print(f"Agent {i} bids truthfully on {truthful}/{num_points} values")

    df = pd.DataFrame({'loss_act': loss_act_list[0], 'loss_q': loss_q_list[0], 'reward': reward_list[0]})
    df.to_csv("./BanditSupervised.csv",
              index=False, header=True)
