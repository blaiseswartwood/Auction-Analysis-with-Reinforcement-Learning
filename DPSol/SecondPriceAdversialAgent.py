import numpy as np
from tqdm import tqdm

# Q Learning for second price auction with two agents and one item for a discrete environment

# Values -> states
# Actions -> bids placed
# Rewards -> from second price auction

class Policy:
    def __init__(self, n_state, n_action, eps, gamma = 0.0, alpha = 0.01):
        self.Q = np.zeros((n_state, n_action))
        self.n_state = n_state
        self.n_action = n_action
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

    def noisy_agent(self):
        return 2

        
    def cheating_agent(self, bids_action, n_identical_items):
        sorted_bids = np.sort(bids_action)[::-1]
        # get the k+1th value and bid just one higher to ensure selection
        if sorted_bids[n_identical_items] == n_action-1:
            return sorted_bids[n_identical_items]
        return sorted_bids[n_identical_items] + 1

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
    n_agents = 7
    n_identical_items = 2

    agents = [Policy(n_state, n_action, 1.0) for x in range(n_agents)]
    auction = Auction(n_state, n_action, n_agents, n_identical_items)
    
    for i in tqdm(range(n_episodes)):                
        # Pick random states aka values to use
        values_state = [np.random.randint(n_state) for x in range(n_agents)]

        # Pick the action based on the state/value
        bids_action = [agents[x].select_action(values_state[x]) for x in range(n_agents-2)]
        bids_action.append(agents[n_agents-2].noisy_agent())
        bids_action.append(agents[n_agents-1].cheating_agent(bids_action, n_identical_items)) 

        # Obtain the reward from those actions
        rewards = auction.second_price_auction(bids_action, values_state)
        
        # Update Q based on learning
        decay_step = 1 / n_episodes
        for x in range(n_agents):
            agents[x].update(values_state[x], bids_action[x], rewards[x])
            agents[x].eps = max(0.001, agents[x].eps - decay_step)

    for x in range(n_agents):

        print(f"\nValue (State) -> Bid (Action) [Agent {x}]")
        truthful_matches = 0
        for state in range(n_state):
            best_action = np.argmax(agents[x].Q[state])
            print(f"Value: {auction.values_states[state]:.2f} → Bid: {auction.actions_bids[best_action]:.2f}")
            # Compare bid to value (rounded to index level)
            if best_action == state:
                truthful_matches += 1
        if x == n_agents-2:
            print("Adversarial Noisy Agent")
        elif x == n_agents-1:
            print("Cheating Agent")
        print(f"\nAgent {x} bids truthfully in {truthful_matches} out of {n_state} states.")

