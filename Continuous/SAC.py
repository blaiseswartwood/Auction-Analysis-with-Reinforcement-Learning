import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import random
from collections import deque
import copy
import pandas as pd
from tqdm import tqdm
import numpy as np

from BidTracker import BidValueTracker
from utils import select_mode, parse_args, Auction

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SAC for continuous environments
# Values -> states
# Actions -> bids placed
# Rewards -> from second price auction

class SAC:
    def __init__(self, act_lim_high, act_lim_low, n_state=1, n_action=1, ):
        self.act_lim_high = act_lim_high
        self.act_lim_low = act_lim_low
        self.alpha = 0.01
        self.n_action = n_action
        self.n_state = n_state

        # Actor network
        self.act_net = nn.Sequential(
            nn.Linear(n_state, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 2 * n_action),
        )

        # Critic networks
        self.q1_net = nn.Sequential(
            nn.Linear(n_state + n_action, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(n_state + n_action, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

        self.act_net.to(device)
        self.q1_net.to(device)
        self.q2_net.to(device)

        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=1e-3)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=1e-3)
        self.act_optimizer = torch.optim.Adam(self.act_net.parameters(), lr=1e-3)

    def update(self, state, action, reward):
        # Convert inputs to tensors
        state = torch.FloatTensor(np.atleast_2d(state)).to(device)
        action = torch.FloatTensor(np.atleast_2d(action)).to(device)
        reward = torch.FloatTensor(np.atleast_1d(reward)).to(device)

        # Q-function predictions
        q_input = torch.cat([state, action], dim=1)
        q1_pred = self.q1_net(q_input).squeeze()
        q2_pred = self.q2_net(q_input).squeeze()
        reward = reward.view_as(q1_pred)

        # Critic loss (MSE)
        loss_q1 = F.mse_loss(q1_pred, reward)
        loss_q2 = F.mse_loss(q2_pred, reward)

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        # Actor update
        out = self.act_net(state)
        mu = torch.sigmoid(out[:, :self.n_action])
        var = F.softplus(out[:, self.n_action:]) + 1e-5        
        dist = Normal(mu, var)
        act_sample = dist.rsample()
        logprob = dist.log_prob(act_sample).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()

        q_input_new = torch.cat([state, act_sample], dim=1)
        q1_val = self.q1_net(q_input_new)
        q2_val = self.q2_net(q_input_new)
        q_val = torch.min(q1_val, q2_val).squeeze()

        actor_loss = -(q_val - self.alpha * logprob).mean()

        self.act_optimizer.zero_grad()
        actor_loss.backward()
        self.act_optimizer.step()

        return loss_q1.item(), loss_q2.item(), actor_loss.item(), entropy.item()

    def select_action(self, state):
        state = torch.FloatTensor(np.atleast_1d(state)).to(device)
        with torch.no_grad():
            out = self.act_net(state)
            mu = torch.sigmoid(out[:self.n_action])
            var = F.softplus(out[self.n_action:]) + 1e-5
            dist = Normal(mu, var)
            action = dist.sample().item()
        return np.clip(action, self.act_lim_low, self.act_lim_high)

if __name__ == "__main__":
    # range of continuous actions
    act_lim_high = 1.0
    act_lim_low = 0.0
    
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
    
    agents = [SAC(act_lim_high, act_lim_low) for x in range(n_agents)]
    auction = Auction(n_agents, n_items)

    reward_list = [[] for _ in range(n_agents)]
    loss_q1_list = [[] for _ in range(n_agents)]
    loss_q2_list = [[] for _ in range(n_agents)]
    loss_act_list = [[] for _ in range(n_agents)]
    loss_ent_list = [[] for _ in range(n_agents)]

    tracker = BidValueTracker(n_agents=n_agents)


    ## -- Training Loop -- ##
    for episode in tqdm(range(n_episodes)):
        # Generate random values (states)
        values = [np.random.uniform(0, 1) for _ in range(n_agents)]

        # Get bids from agents
        if args.mode == "4" or args.mode == "multi-k-adversial":
            bids = [agents[i].select_action([values[i]]) for i in range(n_agents-2)]
            bids.append(agents[-2].noisy_agent())
            bids.append(agents[-1].cheating_agent(bids, n_items))
        else:
            bids = [agents[i].select_action([values[i]]) for i in range(n_agents)]

        # Rewards from auction mechanism
        rewards = auction.second_price_auction(bids, values)

        # Update agents
        for i in range(n_agents):
            loss_q1, loss_q2, loss_act, loss_ent = agents[i].update([values[i]], bids[i], rewards[i])
            # Recording
            if episode % record_freq == 0:
                loss_q1_list[i].append(loss_q1)
                loss_q2_list[i].append(loss_q2)
                loss_act_list[i].append(loss_act)
                loss_ent_list[i].append(loss_ent)
                reward_list[i].append(rewards[i])
                tracker.capture_checkpoint(agents, episode, algorithm_type="sac")


    ## -- Results -- ##
    # Print learned bidding policy
    for i, agent in enumerate(agents):
        print(f"\nAgent {i} bidding policy:")
        truthful = 0
        for v in np.linspace(0, 1, num_points):
            bid = agent.select_action([v])
            print(f"Value: {v:.2f} → Bid: {bid:.2f}")
            if abs(bid - v) <= truth_max:
                truthful += 1
        print(f"Agent {i} bids truthfully in {truthful}/{num_points} sampled values.")
        if args.mode == "4" or args.mode == "multi-k-adversial":
            if x == n_agents-2:
                print("Adversarial Noisy Agent")
            elif x == n_agents-1:
                print("Cheating Agent")
                
    df = pd.DataFrame({'loss_act': loss_act_list[0], 'loss_q1': loss_q1_list[0], 'loss_q2': loss_q2_list[0], 'loss_ent': loss_ent_list[0], 'reward': reward_list[0]})
    df.to_csv("./SAC.csv",
              index=False, header=True)
    tracker.create_plots(algorithm_type="sac")
