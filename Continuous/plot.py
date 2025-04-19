import pandas as pd
from pandas.core.indexes.base import Index
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
            df["tag"] = "".join(entry.name.split(".")[:-1])
            df[y] = df[y].rolling(smooth).mean()
            dfs.append(df)

    all_df = pd.concat(dfs)
    return all_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="data directory",
                        type=str, default="./Data/")
    parser.add_argument("y", help="which column to plot",
                        type=str)
    parser.add_argument("--smooth", "-s", help="moving average len",
                        nargs='?', default=1,
                        type=int)
    parser.add_argument("--out", "-o", help="output filename (PNG)",
                        type=str, default="plot.png")
    args = parser.parse_args()

    fig, ax = plt.subplots()
    df = read_data(args.path, args.y, args.smooth)
    sns.lineplot(data=df, x=df.index,
                 y=args.y, hue="tag", style="tag")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    # Save the figure
    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")

    plt.show()
