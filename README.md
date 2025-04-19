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

```bash
python file_to_run.py
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
