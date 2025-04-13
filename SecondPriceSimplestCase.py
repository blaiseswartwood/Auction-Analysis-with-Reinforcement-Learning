import numpy as np
from tqdm import tqdm

# Q Learning for second price auction with two agents and one item for a discrete environment

# Values -> states
# Actions -> bids placed
# Rewards -> from second price auction

class Policy:
    def __init__(self, Q, eps, gamma = 0.0, alpha = 0.01):
        self.Q = Q
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def update(self, state, action, reward, next_state = None):
        # since this is a stateless environment (one round played til the end, there is no 'next state' to consider)
        # typically target = self.gamma * np.max(self.Q[next_state])
        target = self.gamma * np.max(self.Q[state])
        self.Q[state][action] += self.alpha * (reward + target - self.Q[state][action])

    # epsilon greedy
    def select_action(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(len(self.Q[state]))
        else:
            return np.argmax(self.Q[state])

class Auction:
    def __init__(self, n_state, n_action):
        self.values_states = np.linspace(0, 1, n_state)
        self.actions_bids = np.linspace(0, 1, n_action)

    # define reward as internal value - price paid
    def second_price_auction(self, bid1, bid2, val1, val2):
        # working entirely based on indexes until here

        # agent1 wins
        if bid1 > bid2:
            return self.values_states[val1] - self.actions_bids[bid2], 0
        elif bid2 > bid1: # agent2 wins
            return 0, self.values_states[val2] - self.actions_bids[bid1]
        else:
            if np.random.rand() < 0.5:
                return self.values_states[val1] - self.actions_bids[bid2], 0
            else:
                return 0, self.values_states[val2] - self.actions_bids[bid1]

if __name__ == "__main__":
    n_state = 11
    n_action = 11
    n_episodes = 1000000    
    Q1 = np.zeros((n_state, n_action))
    Q2 = np.zeros((n_state, n_action))

    agent1 = Policy(Q1, 1.0)
    agent2 = Policy(Q2, 1.0)

    auction = Auction(n_state, n_action)
    
    for i in tqdm(range(n_episodes)):                
        # Pick random states aka values to use
        value1_state = np.random.randint(n_state)
        value2_state = np.random.randint(n_state)

        # Pick the action based on the state/value
        bid1_action = agent1.select_action(value1_state)
        bid2_action = agent2.select_action(value2_state)

        # Obtain the reward from those actions
        r1, r2 = auction.second_price_auction(bid1_action, bid2_action, value1_state, value2_state)
        
        # Update Q based on learning
        agent1.update(value1_state, bid1_action, r1)
        agent2.update(value2_state, bid2_action, r2)
            
        # reduce exploration overtime
        decay_step = 1.0 / n_episodes
        agent1.eps = max(0.01, agent1.eps - decay_step)
        agent2.eps = max(0.01, agent2.eps - decay_step)
    
    print("\nValue (State) -> Bid (Action) [Agent 1]")
    truthful_matches = 0
    for state in range(n_state):
        best_action = np.argmax(agent1.Q[state])
        print(f"Value: {auction.values_states[state]:.2f} → Bid: {auction.actions_bids[best_action]:.2f}")
        # Compare bid to value (rounded to index level)
        if best_action == state:
            truthful_matches += 1
    
    print(f"\nAgent 1 bids truthfully in {truthful_matches} out of {n_state} states.")

    print("\nValue (State) -> Bid (Action) [Agent 2]")
    truthful_matches = 0
    for state in range(n_state):
        best_action = np.argmax(agent2.Q[state])
        print(f"Value: {auction.values_states[state]:.2f} → Bid: {auction.actions_bids[best_action]:.2f}")
        # Compare bid to value (rounded to index level)
        if best_action == state:
            truthful_matches += 1
    
    print(f"\nAgent 2 bids truthfully in {truthful_matches} out of {n_state} states.")
