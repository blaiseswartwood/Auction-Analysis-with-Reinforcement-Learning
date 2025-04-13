import numpy as np
from tqdm import tqdm

# Q Learning for second price auction with two agents and one item for a discrete environment

# Values -> states
# Actions -> bids placed
# Rewards -> from second price auction

class Policy:
    def __init__(self, Q, eps, gamma = 0.0, alpha = 0.01):
        self.Q = np.zeros((n_state, n_action))
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
    def __init__(self, n_state, n_action, n_agent, n_identical_items):
        self.values_states = np.linspace(0, 1, n_state)
        self.actions_bids = np.linspace(0, 1, n_action)
        self.n_agent = n_agent
        self.n_identical_items = n_identical_items

    def second_price_auction(self, bids, values):
        rewards = np.zeros(self.n_agent)
    
        bid_amounts = self.actions_bids[bids]
        value_amounts = self.values_states[values]
    
        # Tie-breaking by shuffling before sort
        agent_indices = np.arange(len(bid_amounts))
        np.random.shuffle(agent_indices)

        # bid_amounts[agent_indicies] -> get bid amounts after shuffle
        # np.argsort -> get back indicies to sort in descending order
        # rearrange agent_indicies to match 
        sorted_indices = agent_indices[np.argsort(bid_amounts[agent_indices])[::-1]]
        k_plus_1_price = bid_amounts[sorted_indices[self.n_identical_items]]

        for x in range(self.n_identical_items):
            winner = sorted_indices[x]
            reward = value_amounts[winner] - k_plus_1_price
            rewards[winner] = reward
    
        return rewards

if __name__ == "__main__":
    n_state = 11
    n_action = 11
    n_episodes = 1000000    
    n_agents = 5
    n_identical_items = 2

    agents = [Policy(n_state, n_action, 1.0) for x in range(n_agents)]
    auction = Auction(n_state, n_action, n_agents, n_identical_items)
    
    for i in tqdm(range(n_episodes)):                
        # Pick random states aka values to use
        values_state = [np.random.randint(n_state) for x in range(n_agents)]

        # Pick the action based on the state/value
        bids_action = [agents[x].select_action(values_state[x]) for x in range(n_agents)]

        # Obtain the reward from those actions
        rewards = auction.second_price_auction(bids_action, values_state)
        
        # Update Q based on learning
        decay_step = 1.0 / n_episodes
        for x in range(n_agents):
            agents[x].update(values_state[x], bids_action[x], rewards[x])
            agents[x].eps = max(0.01, agents[x].eps - decay_step)

    for x in range(n_agents):
        print(f"\nValue (State) -> Bid (Action) [Agent {x}]")
        truthful_matches = 0
        for state in range(n_state):
            best_action = np.argmax(agents[x].Q[state])
            print(f"Value: {auction.values_states[state]:.2f} → Bid: {auction.actions_bids[best_action]:.2f}")
            # Compare bid to value (rounded to index level)
            if best_action == state:
                truthful_matches += 1
        
        print(f"\nAgent {x} bids truthfully in {truthful_matches} out of {n_state} states.")

