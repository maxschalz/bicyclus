"""Merge PyMC traces stored as CDF files and plot traces."""

import argparse
import getpass
import glob
import json
from os import path
import re
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from bicyclus.util import log


def merge_samples(files, dim):
    samples = []
    for f in files:
        samples.append(az.InferenceData.from_netcdf(f))
    return az.concat(samples, dim=dim)


def readcdf(filename):
    return az.InferenceData.from_netcdf(filename)


def prepare_df(trace):
    df = trace.to_dataframe()
    renames = {c: c[1] for c in df.columns if type(c) is tuple and c[0] == "posterior"}
    df.rename(columns=renames, inplace=True)
    return df


def plot_all_histograms(jobid, format, trace):
    log.log_print(f"Plotting all histograms in array for job {jobid}")
    df = prepare_df(trace)
    g = sb.PairGrid(data=df, vars=df.columns[2:], height=4)
    g.map_upper(sb.scatterplot)
    g.map_lower(sb.kdeplot)
    g.map_diag(sb.histplot)
    g.savefig(f"seaborn_all_hists_{jobid}.{format}")


def plot_samples(trace_samples, draw_samples, plotfile, densfile, combined=True):
    log.log_print("Plotting trace")
    az.plot_trace(trace_samples, combined=combined)
    plt.savefig(plotfile)

    log.log_print("Plotting density")
    az.plot_density(
        draw_samples,
        grid=(len(draw_samples.posterior), 1),
        figsize=(10, len(draw_samples.posterior) * 5),
        hdi_prob=1.0,
    )
    plt.savefig(densfile)


def plot_hist2d(hist_vars, samples, histfile, kind="kde"):
    """Plot two variables as 2D histogram.

    Arguments:
        hist_vars: list of str
        samples: az.InferenceData, joined by dim=draw
        histfile: destination file name for plot.
        kind: 'mpl' or one of seaborn's { 'scatter' | 'kde' | 'hist' | 'hex' | 'reg' | 'resid' }
    """
    if len(samples.posterior) == 2:
        var1, var2 = samples.posterior
    else:
        var1, var2 = hist_vars
    log.log_print("Histogramming variables:", [var1, var2])

    # ignore values lower or higher than 10th or 100th percentile (lower values are often not as important)
    lo, hi = 0.005, 0.995

    x, y = np.array(samples.posterior[var1]), np.array(samples.posterior[var2])
    x = x.reshape(x.size)
    y = y.reshape(y.size)
    mask = (np.isnan(x)) | (np.isnan(y))
    x = x[~mask]
    y = y[~mask]

    xrange = (np.quantile(x, lo), np.quantile(x, hi))
    yrange = (np.quantile(y, lo), np.quantile(y, hi))
    log.log_print("ranges x/y:", xrange, yrange)

    if kind == "mpl":
        plt.figure(figsize=(12, 10))
        plt.tight_layout()
        h = plt.hist2d(x, y, bins=50, range=(xrange, yrange))
        plt.colorbar(h[3])
        plt.xlabel(var1)
        plt.ylabel(var2)
    else:
        df = samples.to_dataframe()
        df = df.rename(columns={("posterior", v): v for v in [var1, var2]})
        sb.jointplot(
            data=df, x=var1, y=var2, kind=kind, xlim=xrange, ylim=yrange, height=10
        )
    plt.savefig(histfile)


def enumerate_files_for_jobid(tracedir, jobid, typ="cdf"):
    pattern = path.join(tracedir, f"*_{jobid}_*.{typ}")
    candidates = sorted(glob.glob(pattern))
    by_task = {}
    if typ == "cdf":
        task_iter_re = re.compile(f"{jobid}_(\\d+)_(\\d+).{typ}")
    elif typ == "json":
        # JSON sample logs don't have an iteration
        task_iter_re = re.compile(f"{jobid}_(\\d+).{typ}")
    else:
        raise Exception(f"Unknown data file type {typ}")

    for c in candidates:
        m = re.search(task_iter_re, c)
        # candidates are sorted, ensuring that only the latest snapshot is stored in the end.
        by_task[int(m.group(1))] = c
    files = list(by_task.values())
    log.log_print(f"Discovered filenames from pattern {pattern}: {files}")
    return files


def read_json_file_to_trace(filename):
    samples = []
    with open(filename, "r") as file:
        while True:
            l = file.readline()
            if l == "":
                break
            j = json.loads(l)
            samples.append(j["parameters"])
    for_trace = {k: np.zeros(len(samples)) for k in samples[0].keys()}
    for i, s in enumerate(samples):
        for k, v in s.items():
            for_trace[k][i] = v
    return az.convert_to_inference_data(for_trace)


def update_filename(filename, jobid):
    if jobid is None or jobid < 0:
        return filename
    # Ensures that we only modify the last part before the file extension
    parts = filename.split(".")
    parts[-2] = parts[-2] + f"_{jobid}"
    return ".".join(parts)


def run(args):
    files = args.infiles

    if not args.json:
        if not files:
            files = enumerate_files_for_jobid(args.trace_dir, args.jobid)
        trace_samples = merge_samples(files, "chain")
        draw_samples = merge_samples(files, "draw")

        log.log_print(trace_samples.posterior)
        log.log_print(
            "Available posterior distributions:", list(trace_samples.posterior)
        )
        print(trace_samples.posterior)
        print(az.summary(trace_samples))

        plot_samples(
            trace_samples if args.dim == "chain" else draw_samples,
            draw_samples,
            update_filename(
                args.trace_plot_file.format(format=args.format), args.jobid
            ),
            update_filename(
                args.density_plot_file.format(format=args.format), args.jobid
            ),
            args.combined,
        )
        plot_all_histograms(args.jobid or 0, args.format, trace_samples)
        if len(draw_samples.posterior) == 2 or args.hist_vars is not None:
            plot_hist2d(
                args.hist_vars.split(",") if args.hist_vars else [],
                draw_samples,
                update_filename(
                    args.histogram_plot_file.format(format=args.format), args.jobid
                ),
                kind=args.hist_kind,
            )
        if args.outcdf:
            trace_samples.to_netcdf(update_filename(args.outcdf, args.jobid))
    else:  # args.json
        if not files:
            files = enumerate_files_for_jobid(args.trace_dir, args.jobid, typ="json")
        traces = [read_json_file_to_trace(f) for f in files]
        merged = az.concat(traces, dim="draw")
        log.log_print("Available posterior distributions:", list(merged.posterior))
        log.log_print("Using JSON traces: merging along `draw`")
        print(merged.posterior)
        print(az.summary(merged))
        plot_samples(
            merged,
            merged,
            update_filename(
                args.trace_plot_file.format(format=args.format), args.jobid
            ),
            update_filename(
                args.density_plot_file.format(format=args.format), args.jobid
            ),
            args.combined,
        )
        if args.outcdf:
            merged.to_netcdf(update_filename(args.outcdf, args.jobid))


def main():
    user = getpass.getuser()
    p = argparse.ArgumentParser(
        description="Cluster sampling test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "infiles",
        type=str,
        default=None,
        nargs="*",
        help="CDF files for merging (or use --trace-dir)",
    )
    p.add_argument(
        "--trace-dir",
        type=str,
        default=f"/work/{user}/data/",
        help="Directory to search for traces in",
    )
    p.add_argument(
        "--jobid",
        type=int,
        default=-1,
        help="Job ID whose traces to process. This expects CDF files of the pattern: <job name stuff>_<job id>_<task id>_<iteration>.cdf",
    )
    p.add_argument(
        "--json",
        default=False,
        action="store_true",
        help="Analyse samples from JSON logs instead of CDF files.",
    )
    p.add_argument(
        "--trace-plot-file",
        type=str,
        default="plot_merge_trace.{format}",
        help="Plot output for trace plot",
    )
    p.add_argument(
        "--density-plot-file",
        type=str,
        default="plot_merge_density.{format}",
        help="Plot output for density plot",
    )
    p.add_argument(
        "--histogram-plot-file",
        type=str,
        default="plot_merge_histogram.{format}",
        help="Plot output for histogram plot if there are two posteriors",
    )
    p.add_argument(
        "--hist-kind",
        type=str,
        default="kde",
        help="Type of 2D histogram if selected. One of mpl (for matplotlib hist2d), kde/scatter/hist/hex/reg/resid",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=("pdf", "png"),
        default="png",
        help="Output format for plots",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Combine results of all chains in one graph line",
    )
    p.add_argument(
        "--no-combined",
        dest="combined",
        action="store_false",
        help="Combine results of all chains in one graph line",
    )
    p.add_argument(
        "--outcdf",
        type=str,
        default="merged.cdf",
        help="File to write merged traces to.",
    )
    p.add_argument(
        "--hist-vars",
        type=str,
        default=None,
        help='If there are more than two variables available, select which ones to use for plotting a 2D histogram. E.g., "enrichment,cycle_time"',
    )
    p.add_argument(
        "--dim",
        type=str,
        default="chain",
        help="Whether to merge along `chain` or `draw` (https://arviz-devs.github.io/arviz/api/generated/arviz.concat.html) for the trace plot.",
    )
    args = p.parse_args()

    log.log_print("running with", args)
    az.style.use("arviz-darkgrid")
    run(args)


if __name__ == "__main__":
    main()
