import argparse
import sys
from traceback import print_exc

import matplotlib.pyplot as plt
import arviz as az


def list_parameters(path):
    trace = az.InferenceData.from_netcdf(path)
    print("=== SUMMARY ===")
    print(az.summary(trace))


def plot(args):
    trace = az.InferenceData.from_netcdf(args.input_cdf)
    summary = az.summary(trace)
    tracedf = trace.to_dataframe()

    if args.x is not None and args.y is not None:
        if args.x not in summary.index or args.y not in summary.index:
            msg = (
                f"Could not find keys {(args.x, args.y)}. "
                f"Keys in summary: {summary.index}"
            )
            raise KeyError(msg)

        try:
            xs = tracedf[("posterior", args.x)]
            ys = tracedf[("posterior", args.y)]
        except KeyError as e:
            print(
                f"ERROR: Could not find keys ('posterior', {args.x}) and "
                f"('posterior', {args.y}) in {args.input_cdf}.\n"
                f"Retrying with keys {args.x} and {args.y}."
            )

        xs = tracedf[args.x]
        ys = tracedf[args.y]

        xvar = args.x
        yvar = args.y
    else:
        variables = summary.index
        if len(variables) > 2:
            raise RuntimeError(f"Too many variables: {variables}")
        elif len(variables) < 2:
            raise RuntimeError(f"Only one variable in tracedf: {variables}")
        xs = tracedf[("posterior", variables[0])]
        ys = tracedf[("posterior", variables[1])]
        (xvar, yvar) = variables

    assert len(xs) == len(ys)
    if len(xs) > args.first:
        xs = xs[0 : args.first]
        ys = ys[0 : args.first]

    plt.figure(figsize=(10, 10), constrained_layout=True)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    if not args.hist:
        plt.plot(xs.to_numpy(), ys.to_numpy(), linewidth=0.5)
    else:
        plt.hist2d(xs.to_numpy(), ys.to_numpy(), bins=50)
    plt.savefig(args.out)


def main():
    p = argparse.ArgumentParser(description="visualize a 2D walk along distribution")
    p.add_argument(
        "input_cdf", type=str, default=None, help="Input file containing traces"
    )
    p.add_argument("--x", type=str, default=None, help="Parameter space: x dimension")
    p.add_argument("--y", type=str, default=None, help="Parameter space: y dimension")
    p.add_argument(
        "-l",
        action="store_const",
        const=True,
        default=False,
        help="List available parameters by name",
    )
    p.add_argument(
        "--out", type=str, default="walk2d.png", help="Destination file for plot"
    )
    p.add_argument(
        "--first", type=int, default=-1, help="Only consider first X samples"
    )
    p.add_argument(
        "--hist",
        action="store_const",
        const=True,
        default=False,
        help="Plot histogram instead of walk",
    )
    args = p.parse_args()

    if args.l:
        list_parameters(args.input_cdf)
        return

    plot(args)


if __name__ == "__main__":
    main()
