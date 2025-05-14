import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from comb_dp_utils import select_mode, parse_args

# Q Learning for VCG combinatorial auction with additive valuations
# Values for items -> states
# Actions -> bids placed for items
# Rewards -> from VCG mechanism

class Policy:
    def __init__(self, n_state, n_action, n_items, eps, gamma=0.0, alpha=0.01):
        self.Q = np.zeros((n_items, n_state, n_action)) #For additive valuations, we keep a separate Q table for each item
        self.n_state = n_state
        self.n_action = n_action
        self.n_items = n_items
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def update(self, states, actions, reward):
        # Update Q values for each item separately
        # The reward is the total reward from the auction
        # We distribute it proportionally to items based on value
        total_value = sum(self.values_states[states[i]] for i in range(self.n_items))
        
        for item in range(self.n_items):
            state = states[item]
            action = actions[item]
            #proportional reward allocation
            item_reward = reward * (self.values_states[state] / total_value) if total_value > 0 else reward / self.n_items
            
            #same idea as before
            target = self.gamma * np.max(self.Q[item][state])
            self.Q[item][state][action] += self.alpha * (item_reward + target - self.Q[item][state][action])

    # epsilon greedy, altered to work for multiple items
    def select_actions(self, states):
        actions = []
        for item in range(self.n_items):
            state = states[item]
            if np.random.rand() < self.eps:
                actions.append(np.random.randint(self.n_action))
            else:
                actions.append(np.argmax(self.Q[item][state]))
        return actions

    def noisy_agent(self, states):
        return [np.random.randint(self.n_action) for _ in range(self.n_items)]
        
    def cheating_agent(self, all_bids, states):
        # A more sophisticated cheating strategy for VCG
        # This agent tries to bid just enough to win items it values highly
        cheating_bids = []
        for item in range(self.n_items):
            # Get all bids for this item from other agents
            item_bids = [all_bids[agent][item] for agent in range(len(all_bids))]
            # Get the highest bid
            highest_bid = max(item_bids)
            # Bid slightly above if the item is valuable enough
            if self.values_states[states[item]] > self.actions_bids[highest_bid]:
                cheating_bids.append(min(highest_bid + 1, self.n_action - 1))
            else:
                # Bid truthfully for less valuable items
                cheating_bids.append(states[item])
        return cheating_bids


class Auction:
    def __init__(self, n_state, n_action, n_agent, n_items):
        self.values_states = np.linspace(0, 1, n_state)
        self.actions_bids = np.linspace(0, 1, n_action)
        self.n_agent = n_agent
        self.n_items = n_items
        
        # Make these accessible to the Policy class
        Policy.values_states = self.values_states
        Policy.actions_bids = self.actions_bids

    def vcg_auction(self, bids, values):
        """
        Implement VCG mechanism for combinatorial auction with additive valuations
        
        Parameters:
        - bids: list of lists, where bids[agent][item] is the action index for agent's bid on item
        - values: list of lists, where values[agent][item] is the state index for agent's value for item
        
        Returns:
        - rewards: array of utilities (value - payment) for each agent
        """
        rewards = np.zeros(self.n_agent)
    
        bid_values = []
        true_values = []
        
        for agent in range(self.n_agent):
            bid_values.append([self.actions_bids[bids[agent][item]] for item in range(self.n_items)])
            true_values.append([self.values_states[values[agent][item]] for item in range(self.n_items)])
        
        #find efficient allocation (max social welfare based on bids)
        allocation = self._find_efficient_allocation(bid_values)
        
        #calculate VCG payments for each agent
        payments = self._calculate_vcg_payments(bid_values, allocation)
        
        #calculate rewards (utility = value - payment)
        for agent in range(self.n_agent):
            agent_value = sum(true_values[agent][item] for item in range(self.n_items) 
                             if allocation[agent][item])
            rewards[agent] = agent_value - payments[agent]
        
        return rewards
    
    def _find_efficient_allocation(self, bid_values):
        #Find allocation that maximizes social welfare based on reported bids
        allocation = [[False for _ in range(self.n_items)] for _ in range(self.n_agent)]
        
        #assign each item to highest bidder, value is additive
        for item in range(self.n_items):

            item_bids = [bid_values[agent][item] for agent in range(self.n_agent)] #all the bids
            
            max_bid = max(item_bids)
            #in case of ties
            max_bidders = [agent for agent, bid in enumerate(item_bids) if bid == max_bid] # TODO: make this more efficient. Is there a iloc method?
            winner = np.random.choice(max_bidders)
            
            #allocate
            allocation[winner][item] = True
            
        return allocation
    
    def _calculate_vcg_payments(self, bid_values, allocation):
        payments = [0 for _ in range(self.n_agent)]
        
        for agent in range(self.n_agent):
            # Calculate welfare of others with agent
            others_welfare_with = 0
            for other in range(self.n_agent):
                if other != agent:
                    others_welfare_with += sum(bid_values[other][item] for item in range(self.n_items)
                                            if allocation[other][item])
            
            # Find counterfactual allocation without agent
            cf_allocation = self._find_counterfactual_allocation(bid_values, agent)
            
            # Calculate welfare of others without agent
            others_welfare_without = 0
            for other in range(self.n_agent):
                if other != agent:
                    others_welfare_without += sum(bid_values[other][item] for item in range(self.n_items)
                                               if cf_allocation[other][item])
            
            # VCG payment is the welfare difference
            payments[agent] = others_welfare_without - others_welfare_with
            
        return payments
    
    def _find_counterfactual_allocation(self, bid_values, excluded_agent):
        cf_allocation = [[False for _ in range(self.n_items)] for _ in range(self.n_agent)]
        
        # Similar to _find_efficient_allocation but exclude the given agent
        for item in range(self.n_items):
            # Get bids excluding the excluded agent (set their bid to -inf)
            item_bids = []
            for agent in range(self.n_agent):
                if agent == excluded_agent:
                    item_bids.append(float('-inf'))
                else:
                    item_bids.append(bid_values[agent][item])
            
            # Find winner among remaining agents
            max_bid = max(item_bids)
            if max_bid > float('-inf'):  # Ensure at least one valid bid
                max_bidders = [agent for agent, bid in enumerate(item_bids) if bid == max_bid]
                winner = np.random.choice(max_bidders)
                cf_allocation[winner][item] = True
                
        return cf_allocation


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
    n_state = config["n_state"]
    n_action = config["n_action"]

    print(f"Mode: {args.mode}")
    print(f"Agents: {n_agents}, Items: {n_items}, Episodes: {n_episodes}, States: {n_state}, Actions: {n_action}")

    # Create agents (policies)
    agents = [Policy(n_state, n_action, n_items, 1.0) for _ in range(n_agents)]
    auction = Auction(n_state, n_action, n_agents, n_items)
    for i in tqdm(range(n_episodes)):                
        #generate random values for each item
        values_states = [[np.random.randint(n_state) for _ in range(n_items)] for _ in range(n_agents)]

        if args.mode == "4" or args.mode == "multi-k-adversial":
            bids_actions = [agents[x].select_actions(values_states[x]) for x in range(n_agents-2)] 
            bids_actions.append(agents[n_agents-2].noisy_agent(values_states[n_agents-2])) 
            bids_actions.append(agents[n_agents-1].cheating_agent(bids_actions, values_states[n_agents-1]))
        else:
            bids_actions = [agents[x].select_actions(values_states[x]) for x in range(n_agents)]

        rewards = auction.vcg_auction(bids_actions, values_states) #(replaced second price with VCG)
        
        decay_step = 1 / n_episodes
        for x in range(n_agents):
            agents[x].update(values_states[x], bids_actions[x], rewards[x])
            agents[x].eps = max(0.001, agents[x].eps - decay_step)

    results = []
    #print results
    for agent_idx in range(n_agents):
        print(f"\nAgent {agent_idx} bidding strategy:")
        
        truthful_matches = 0
        total_states = 0
        
        for item in range(n_items):
            print(f"\nItem {item}:")
            print("Value (State) -> Bid (Action)")
            
            for state in range(n_state):
                best_action = np.argmax(agents[agent_idx].Q[item][state])
                print(f"Value: {auction.values_states[state]:.2f} → Bid: {auction.actions_bids[best_action]:.2f}")
                
                if np.abs(best_action - state) <= 1:  # Allow slight deviation
                    truthful_matches += 1
                total_states += 1

                #plotting data
                value = auction.values_states[state]
                bid = auction.actions_bids[best_action]
                diff = bid - value  # positive = overbid, negative = underbid
                results.append({
                "Agent": x,
                "State Index": state,
                "Value": value,
                "Best Bid": bid,
                "Difference": diff
            })
        
        if args.mode == "4" or args.mode == "multi-k-adversial":
            if agent_idx == n_agents-2:
                print("Adversarial Noisy Agent")
            elif agent_idx == n_agents-1:
                print("Cheating Agent")
                
        truthful_percentage = (truthful_matches / total_states) * 100
        print(f"\nAgent {agent_idx} bids truthfully in {truthful_matches} out of {total_states} states ({truthful_percentage:.1f}%).")

       

    results = pd.DataFrame(results)
    print('plotting')
    #plot
    for x in range(n_agents):
        agent_df = results[results["Agent"] == x]
        plt.plot(agent_df["State Index"], agent_df["Difference"], label=f"Agent {x}")