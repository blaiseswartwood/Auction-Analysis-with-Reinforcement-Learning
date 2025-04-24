# This code was taken and adapted from CSSE490 Deep Reinforcement Learning from Rose-Hulman

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

sns.set_theme(style="darkgrid")

def read_data(dir_path, y, smooth):
    dfs = []
    for entry in os.scandir(dir_path):
        if entry.name.endswith(".csv"):
            df = pd.read_csv(entry, index_col=None)
            df["episode"] = range(len(df))  # Explicit episode index
            df["tag"] = os.path.splitext(entry.name)[0]  # Use file name without extension
            df[y] = df[y].rolling(smooth, min_periods=1).mean()  # Apply moving average
            dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory containing CSV files", type=str)
    parser.add_argument("y", help="Column name to plot on y-axis", type=str)
    parser.add_argument("--smooth", "-s", help="Moving average window length", type=int, default=1)
    parser.add_argument("--out", "-o", help="Output filename for plot (PNG)", type=str, default="plot.png")
    args = parser.parse_args()

    fig, ax = plt.subplots()
    df = read_data(args.path, args.y, args.smooth)

    sns.lineplot(data=df, x="episode", y=args.y, hue="tag", style="tag", ax=ax)

    ax.set_xlabel("recorded episode")
    ax.set_ylabel(args.y)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")
    plt.show()
