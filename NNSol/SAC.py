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

# SAC for continuous environments
# Values -> states
# Actions -> bids placed
# Rewards -> from second price auction
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAC:
    def __init__(self, n_state, n_action, act_lim_high, act_lim_low):
        self.act_lim_high = act_lim_high
        self.act_lim_low = act_lim_low
        self.alpha = 0.01
        self.n_action = n_action  # save for later use

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

class Auction:
    def __init__(self, n_state, n_action, n_agent, n_identical_items):
        self.n_agent = n_agent
        self.n_identical_items = n_identical_items

    def second_price_auction(self, bids, values):
        rewards = np.zeros(self.n_agent) 

        bids = np.array(bids)
        agent_indices = np.arange(len(bids))
        np.random.shuffle(agent_indices)

        sorted_indices = agent_indices[np.argsort(bids[agent_indices])[::-1]]
        k_plus_1_price = bids[sorted_indices[self.n_identical_items]]

        for x in range(self.n_identical_items):
            winner = sorted_indices[x]
            reward = values[winner] - k_plus_1_price
            rewards[winner] = reward
        return rewards

if __name__ == "__main__":
    # dimension of the state and actions
    n_state = 1
    n_action = 1

    # range of continuous actions
    act_lim_high = 1.0
    act_lim_low = 0.0
    
    n_episodes = 20000    
    n_agents = 2
    n_identical_items = 1
    record_freq = n_episodes/500
    truth_max = 0.05
    num_points = 100
    
    agents = [SAC(n_state, n_action, act_lim_high, act_lim_low) for x in range(n_agents)]
    auction = Auction(n_state, n_action, n_agents, n_identical_items)

    reward_list = [[] for _ in range(n_agents)]
    loss_q1_list = [[] for _ in range(n_agents)]
    loss_q2_list = [[] for _ in range(n_agents)]
    loss_act_list = [[] for _ in range(n_agents)]
    loss_ent_list = [[] for _ in range(n_agents)]
    
    ## -- Training Loop -- ##
    for episode in tqdm(range(n_episodes)):
        # Generate random values (states)
        values = [np.random.uniform(0, 1) for _ in range(n_agents)]

        # Get bids from agents
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

    df = pd.DataFrame({'loss_act': loss_act_list[0], 'loss_q1': loss_q1_list[0], 'loss_q2': loss_q2_list[0], 'loss_ent': loss_ent_list[0], 'reward': reward_list[0]})
    df.to_csv("./SAC.csv",
              index=False, header=True)