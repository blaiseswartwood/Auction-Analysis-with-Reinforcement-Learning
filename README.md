# Auction-Analysis-with-Reinforcement-Learning
Analysis of Ideal Auctions using Reinforcement Learning

## Setup

1. Create a virtual environment
2. Dependencies:
   
```bash
pip install numpy torch pandas tqdmp
```

## Usage

Note: Achieving convergence can be difficult and it is easy to get stuck in local optimum. Best results will take several trials to obtain.
Regardless of the 'truth count' that is obtained, you can identify the general trend of the bids being close to the actual values and tending to increase over time.

### Running the code

Usage:

```bash
python file_to_run.py
```

Example:

```bash
python Bandit.py
```
Use the `--mode` flag to select a training configuration:

| Mode Name         | Code | Description                        |
|-------------------|------|------------------------------------|
| `simple`          | `1`  | Default mode with 2 agents, 1 item |
| `multi-agent`     | `2`  | Multiple agents, 1 item            |
| `multi-k-item`    | `3`  | Multiple agents, multiple items    |
| `multi-k-adversial` | `4` | Adversarial setting                |


You can override any of the default values using the following optional flags:

| Flag            | Description                      | Example              |
|-----------------|----------------------------------|----------------------|
| `--n_agents`    | Number of agents                 | `--n_agents 5`       |
| `--n_items`     | Number of items                  | `--n_items 3`        |
| `--n_episodes`  | Total training episodes          | `--n_episodes 50000` |
| `--truth_max`   | Max truthfulness exploration     | `--truth_max 0.2`    |
| `--num_points`  | Number of points for evaluation  | `--num_points 150`   |

Example:

```bash
python Bandit.py --mode 3 --n_agents 6 --n_episodes 60000
```

### Plotting

For the continuous (deep RL) code, to use the pyplot.py to graph, ensure you only have one excel file in the specified directory.

Usage:
```bash
python your_script.py ./your_directory/ name_of_data_column --smooth 10 --out name_of_plot.png
```

Example:
```bash
python ./plot.py . reward --smooth 10 --out BanditTReward.png
```
## Results

### Discrete (Dynamic Programming)

1. Second-price auction simplest case (two agents, one item):
![image](https://github.com/user-attachments/assets/1e5c4052-1630-439c-883d-b390b62397d3)


### Continuous (Deep RL)

1. Second-price auction simplest case (two agents, one item):
![image](https://github.com/user-attachments/assets/6b7bd7a9-fc6e-4900-b344-c88659bace58)

Q Loss

Act Loss

Rewards
