"""Generate plots from simulation data stored in additional files."""

import argparse
import collections
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def gen_df(data):
    """Generate pd.dataframe from data"""
    processed = collections.defaultdict(list)

    def iterate_dict(d):
        """Iterate recursively through dict and store entries."""
        for key, val in d.items():
            if isinstance(val, dict):
                iterate_dict(val)
            else:
                processed[key].append(val)

    for elmt in data:
        iterate_dict(elmt)

    return pd.DataFrame(processed)


def run(args):
    with open(args.infile, "r") as f:
        data = json.load(f)

    path, _ = os.path.split(args.infile)
    data = gen_df(data)

    sns.set_theme()
    for k in args.hists:
        sns.histplot(data, x=k, stat="density")
        plt.savefig(os.path.join(path, f"hist_{k}.png"))
        plt.close()

    g = sns.PairGrid(data[args.hists])
    g.map_upper(sns.histplot)
    g.map_lower(sns.kdeplot, fill=True)
    g.map_diag(sns.histplot, kde=True)
    plt.savefig(os.path.join(path, f"grid_logs.png"))
    plt.close()

    return


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--infile",
        type=str,
        default=None,
        help=("Path to the .json file containing processed LOG " "data."),
    )
    p.add_argument(
        "--hists",
        type=str,
        default="WeapongradeUSink,SeparatedPuSink",
        help="Comma-separated list of labes that should be histogrammed.",
    )
    args = p.parse_args()

    if args.hists:
        args.hists = args.hists.split(",")

    run(args)
    return


if __name__ == "__main__":
    main()
