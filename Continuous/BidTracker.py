import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

class BidValueTracker:
    def __init__(self, n_agents, save_dir='./plots'):
        self.n_agents = n_agents
        self.save_dir = save_dir
        self.checkpoint_data = []
        
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def capture_checkpoint(self, agents, episode, algorithm_type="bandit", num_samples=20):
        checkpoint = {
            'episode': episode,
            'agents': []
        }
        
        values = np.linspace(0, 1, num_samples)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for i, agent in enumerate(agents):
            agent_data = {
                'agent_id': i,
                'values': values.tolist(),
                'bids': []
            }
            
            for v in values:
                if algorithm_type == "bandit":
                    bid = agent.actor(torch.tensor([[v]], dtype=torch.float32).to(device)).item()
                else:
                    bid = agent.select_action([v])
                
                agent_data['bids'].append(bid)
            
            checkpoint['agents'].append(agent_data)
        
        self.checkpoint_data.append(checkpoint)
        
    def create_plots(self, algorithm_type="bandit"):
        if not self.checkpoint_data:
            print("No checkpoint data to plot")
            return
        
        self._plot_all_differences(algorithm_type)
        
        self._plot_aggregated_differences(algorithm_type)
        
        self._export_all_data_to_csv(algorithm_type)
        
        print(f"Plots created and saved to {self.save_dir}/:")
        print(f"1. All differences: {algorithm_type}_all_differences.png")
        print(f"2. Mean difference with smoothing: {algorithm_type}_mean_difference_smoothed.png")
        print(f"3. Complete data: {self.save_dir}/{algorithm_type}_all_data.csv")
        
    def _export_all_data_to_csv(self, algorithm_type):       
        all_rows = []
        
        for checkpoint in self.checkpoint_data:
            episode = checkpoint['episode']
            
            for agent_data in checkpoint['agents']:
                agent_id = agent_data['agent_id']
                values = agent_data['values']
                bids = agent_data['bids']
                
                for i, (value, bid) in enumerate(zip(values, bids)):
                    all_rows.append({
                        'episode': episode,
                        'agent_id': agent_id,
                        'value_idx': i,
                        'value': value,
                        'bid': bid,
                        'difference': bid - value
                    })
        
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(self.save_dir, f'{algorithm_type}_all_data.csv')
        df.to_csv(csv_path, index=False)
        
        self._export_aggregated_stats(algorithm_type)
        
    def _export_aggregated_stats(self, algorithm_type):
        stats_rows = []
        
        for checkpoint in self.checkpoint_data:
            episode = checkpoint['episode']
            
            for agent_data in checkpoint['agents']:
                agent_id = agent_data['agent_id']
                values = agent_data['values']
                bids = agent_data['bids']
                
                differences = [b - v for b, v in zip(bids, values)]
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                truthful_count = sum(abs(diff) <= 0.05 for diff in differences)
                truthful_pct = (truthful_count / len(differences)) * 100
                
                stats_rows.append({
                    'episode': episode,
                    'agent_id': agent_id,
                    'mean_difference': mean_diff,
                    'std_difference': std_diff,
                    'truthful_count': truthful_count,
                    'truthful_pct': truthful_pct,
                    'num_samples': len(values)
                })
        
        df = pd.DataFrame(stats_rows)
        csv_path = os.path.join(self.save_dir, f'{algorithm_type}_aggregated_stats.csv')
        df.to_csv(csv_path, index=False)
        
    
   
    def _plot_all_differences(self, algorithm_type):
        plt.figure(figsize=(12, 8))
        
        episodes = [cp['episode'] for cp in self.checkpoint_data]
        
        agent_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        
        for agent_idx in range(self.n_agents):
            if not all(agent_idx < len(cp['agents']) for cp in self.checkpoint_data):
                continue
                
            agent_color = agent_colors[agent_idx % len(agent_colors)]
            
            num_values = len(self.checkpoint_data[0]['agents'][agent_idx]['values'])
            
            for value_idx in range(num_values):
                differences = []
                value = None  # Will be set later
                
                for cp in self.checkpoint_data:
                    agent = cp['agents'][agent_idx]
                    if value_idx < len(agent['values']):
                        value = agent['values'][value_idx]
                        bid = agent['bids'][value_idx]
                        diff = bid - value  #Positive = overbidding, Negative = underbidding
                        differences.append(diff)
                    else:
                        differences.append(None)  # Handle missing data
                
                
                plt.plot(episodes, differences, 'o-', alpha=0.3, linewidth=1, 
                         color=agent_color,
                         label=f"Agent {agent_idx}, Value {value:.2f}" if value_idx == 0 else "")
        
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label="Truthful Bidding")
        
        plt.legend(loc='best')
        
        plt.xlabel('Training Episodes')
        plt.ylabel('Bid - Value Difference')
        plt.title(f'{algorithm_type.upper()} Training - All Bid-Value Differences')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{self.save_dir}/{algorithm_type}_all_differences.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_aggregated_differences(self, algorithm_type):
       
        plt.figure(figsize=(12, 8))
        
        episodes = [cp['episode'] for cp in self.checkpoint_data]
        
        agent_colors = ['blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        
        for agent_idx in range(self.n_agents):
            if not all(agent_idx < len(cp['agents']) for cp in self.checkpoint_data):
                continue
            
            #get the color for this agent
            agent_color = agent_colors[agent_idx % len(agent_colors)]
                
            means = []
            
            for cp in self.checkpoint_data:
                agent = cp['agents'][agent_idx]
                differences = [b - v for v, b in zip(agent['values'], agent['bids'])]
                means.append(np.mean(differences))
            
            #TODO: Remove the line below that plots the raw points because its a bit too much
            plt.scatter(episodes, means, color=agent_color, s=40, alpha=0.6,
                       label=f"Agent {agent_idx} - Raw")
            
            # Apply smoothing using a moving average if we have enough points
            if len(means) >= 3:
                # Apply smoothing using a moving average 
                window_size = min(5, len(means) // 2 + 1)  # Adjust window size based on data points
                weights = np.ones(window_size) / window_size
                
                # Calculate moving average (smoothed line)
                # Use 'valid' mode which may shorten the output
                smoothed = np.convolve(means, weights, mode='valid')
                
                offset = window_size - 1
                x_smoothed = episodes[offset//2:-(offset - offset//2)] if offset > 1 else episodes
                
                x_smoothed = x_smoothed[:len(smoothed)]
                
                plt.plot(x_smoothed, smoothed, '-', linewidth=3, color=agent_color, 
                         label=f"Agent {agent_idx} - Smoothed")
            else:
                plt.plot(episodes, means, '-', linewidth=2, color=agent_color,
                         label=f"Agent {agent_idx} - Connected")
        
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label="Truthful Bidding")
        
        plt.legend(loc='best')
        plt.xlabel('Training Episodes')
        plt.ylabel('Mean Bid - Value Difference')
        plt.title(f'{algorithm_type.upper()} Training - Mean Bid-Value Difference with Smoothing')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(f'{self.save_dir}/{algorithm_type}_mean_difference_smoothed.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

