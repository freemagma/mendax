import pandas as pd
import seaborn as sns
import os
import json
from parse import parse

sns.set_style("whitegrid")


def get_output_params(run_dir):
    raw_log = [
        line.strip() for line in open(os.path.join(run_dir, "output.txt")).readlines()
    ]
    output = []
    for line in raw_log:
        if "Epoch" in line:
            output.append([])
            continue
        r = parse("Batch {batch:d}; Crew Score: {score:f}", line)
        if not r:
            continue
        output[-1].append(r)
    params = json.load(open(os.path.join(run_dir, "params.json")))
    return output, params


def plot_run(run_dir):
    output, params = get_output_params(os.path.join("saves", run_dir))
    rows = []
    for e, epoch in enumerate(output):
        rows.extend(
            {"Score": r["score"], "Batch": params["EPOCH_LENGTH"] * e + r["batch"]}
            for r in epoch
        )
    df = pd.DataFrame(rows)

    plot = sns.lineplot(data=df, x="Batch", y="Score")
    xticks = [0] + [params["EPOCH_LENGTH"] * (e + 1) for e in range(len(output))]
    plot.set_xticks(xticks)
    plot.set_xticklabels(range(len(output) + 1))
    plot.set_xlabel("Epoch")

    fig = plot.get_figure()
    file_name = os.path.join("plots", run_dir + ".png")
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    fig.savefig(file_name)


def plot_grid_search(grid_dir, axes, query=None):
    ax1, ax2 = axes
    rows = []
    for sub in os.scandir(os.path.join("saves", grid_dir)):
        output, params = get_output_params(sub.path)
        params["FINAL_SCORE"] = output[-1][-1]["score"]
        rows.append(params)
    df = pd.DataFrame(rows)
    if query:
        q_str = " & ".join(f"{k} == {v}" for k, v in query.items())
        df = df.query(q_str)

    plot = sns.heatmap(df.pivot(ax1, ax2, "FINAL_SCORE"))

    plot_dir = os.path.join("plots", grid_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    name = f"{ax1[:4]}-{ax2[:4]}"
    if query:
        name += "_" + "-".join(f"{k[:4]}{v}" for k, v in query.items())
    fig = plot.get_figure()
    fig.savefig(os.path.join(plot_dir, f"{name}.png".lower()))


def main():
    # plot_grid_search("2020-11-24_00-18-06_GRID", ("VIEW_CHANCE", "SABOTAGE_CHANCE"))

    # plot_grid_search(
    #     "2020-11-26_22-56-46_GRID",  # grid search folder
    #     ("M", "ROUNDS"),  # two axes to create the plot over
    #     query={
    #         "N": 8,
    #         "SABOTAGE_CHANCE": 0.25,
    #     },  # filters (to make the data-points unique)
    # )

    plot_run(os.path.join("2020-11-26_22-56-46_GRID", "n8_s0.0_m64_r6"))


if __name__ == "__main__":
    main()