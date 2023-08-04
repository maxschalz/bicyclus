"""A collection of functions to manipulate the output of PyMC runs."""

import argparse
import getpass
import json
import os
import re


def extract_from_log(
    log_fname=None,
    log_dir=None,
    job_id=None,
    out_fname=None,
    only_accepted=True,
    extract_array=True,
    log_keywords=["sink_masses", "total_likelihood"],
):
    """Extract simulation details written to log and store in file.

    Parameters
    ----------
    log_fname : str, optional
        Path to log file. Either `log_fname` or `job_id` must be defined.

    log_dir : path, optional
        Path to logs, only used in conjunction with `job_id`. Defaults to
        `/work/$USER/job_output/`

    job_id : int, optional
        Id of the main job, only used in conjunction with `log_dir`.
        Either `log_fname` or `job_id` must be defined.

    out_fname : str, optional
        Name of the output file.

    only_accepted : bool, optional
        If true, only store values from accepted samples.

    extract_array : bool, optional
        If true, extract data from all logs written in this array job (and
        store them in one file).

    log_keywords: list of str
        Words, parameters, ... that should be pasted from log to outfile.
        *NOTE* This is case sensitive and all words in log_keywords must appear
        on the same line in the log file.
    """
    # Set filenames and paths etc.
    if log_fname is None and job_id is None:
        raise ValueError("Both `log_fname` and `job_id` are None.")

    user = getpass.getuser()
    default_dir = os.path.join("/work", user, "job_output")
    if log_dir is None:
        if log_fname is not None:
            log_dir = os.path.dirname(log_fname)
        else:
            log_dir = default_dir if log_dir is None else log_dir

    if job_id is None:
        job_id = re.compile("_(\d+)_\d+\.log").search(log_fname).group(1)

    out_fname = f"extracted_params_{job_id}.json" if out_fname is None else out_fname

    # Perform the extraction.
    data = []
    if extract_array:
        _, _, files = next(os.walk(log_dir))
        job_files = [
            os.path.join(log_dir, f)
            for f in files
            if re.compile(f"_{job_id}_\\d+\\.log").search(f)
        ]
        for f in job_files:
            data.extend(extract_single_log(f, log_keywords, only_accepted))
    else:
        data = data(log_fname, log_keywords, only_accepted)

    if os.path.isfile(out_fname):
        print(f"{out_fname} already exists.")
        ans = ""
        while ans not in ("y", "n"):
            ans = input("Overwrite existing file? (y/n): ").lower()
        if ans == "n":
            print("Exiting without storing the data.")
            return

    with open(out_fname, "w") as f:
        json.dump(data, f)


def extract_single_log(log_fname, log_keywords, only_accepted):
    """Perform the actual extraction from one log."""
    data = []
    mutate_kw = "Mutating: using parameters "
    datetime_separator = " :: "
    sample_accepted = re.compile("sampling: \d+ \d+/\d+")
    with open(log_fname, "r") as f:
        entry = {}
        for line in f:
            if mutate_kw in line:
                entry = {}
                entry["parameters"] = json.loads(
                    line.split(mutate_kw)[1].replace("'", '"')
                )
            elif all(map(lambda x: x in line, log_keywords)):
                entry["values"] = json.loads(
                    line.split(datetime_separator)[1].replace("'", '"')
                )
                if not only_accepted:
                    data.append(entry)
                continue
            if only_accepted and sample_accepted.search(line):
                data.append(entry)

    return data


def main():
    p = argparse.ArgumentParser(
        description="Store log data in JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--log-fname", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--job-id", type=str, default=None)
    p.add_argument("--out-fname", type=str, default=None)
    p.add_argument("--not-only-accepted", action="store_true", default=None)
    p.add_argument("--not-extract-array", action="store_true")
    p.add_argument("--log-keywords", type=str, default="sink_masses,total_likelihood")
    args = p.parse_args()

    if args.log_keywords:
        args.log_keywords = args.log_keywords.split(",")

    extract_from_log(
        log_fname=args.log_fname,
        log_dir=args.log_dir,
        job_id=args.job_id,
        out_fname=args.out_fname,
        only_accepted=not args.not_only_accepted,
        extract_array=not args.not_extract_array,
        log_keywords=args.log_keywords,
    )


if __name__ == "__main__":
    main()
