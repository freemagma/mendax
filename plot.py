import pandas as pd
import seaborn as sns
import os
import json
from parse import parse


def grid_search_df(grid_dir):
    rows = []
    for sub in os.scandir(os.path.join("saves", grid_dir)):
        output = [
            line.strip()
            for line in open(os.path.join(sub.path, "output.txt")).readlines()
        ]
        params = json.load(open(os.path.join(sub.path, "params.json")))
        params["FINAL_SCORE"] = parse("Batch {b}; Crew Score: {s:f}", output[-1])["s"]
        rows.append(params)
    return pd.DataFrame(rows)


def plot_grid_search(grid_dir, axes, query=None):
    ax1, ax2 = axes
    df = grid_search_df(grid_dir)
    if query:
        q_str = " & ".join(f"{k} == {v}" for k, v in query.items())
        df = df.query(q_str)
    plot = sns.heatmap(df.pivot(ax1, ax2, "FINAL_SCORE"))

    plot_dir = os.path.join("plots", grid_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    name = f"{ax1[:4]}-{ax2[:4]}"
    if query:
        name += "_" + "-".join(f"{k}{v}" for k, v in query.items())
    fig = plot.get_figure()
    fig.savefig(os.path.join(plot_dir, f"{name}.png".lower()))


def main():
    # plot_grid_search("2020-11-24_00-18-06_GRID", ("VIEW_CHANCE", "SABOTAGE_CHANCE"))
    plot_grid_search(
        "2020-11-26_22-56-46_GRID",  # grid search folder
        ("M", "ROUNDS"),  # two axes to create the plot over
        query={
            "N": 16,
            "SABOTAGE_CHANCE": 0.5,
        },  # filters (to make the data-points unique)
    )


if __name__ == "__main__":
    main()