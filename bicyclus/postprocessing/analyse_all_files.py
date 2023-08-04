"""Class to analyse a large amount of Cyclus output files."""

import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import localtime, strftime
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpi4py import MPI

from .sqlite_analyser import SqliteAnalyser


@dataclass
class AnalyseAllFiles(ABC):
    """Class to efficiently analyse a large amount of Cyclus output files.

    Parameters
    ----------
    data_path : pathlib.Path or str
        Path to the data directory. Subdirectories will be scanned, as well.

    imgs_path : pathlib.Path or str
        Path where to store the plots. If the directory does not exist, it will
        be created. If it does exist, a message will be displayed. The existing
        directory will not be deleted, however data might get overwritten.

    job_id : int or str, optional
        If indicated, only files that have `job_id` in their filename will be
        considered.

    max_files : int, optional
        If indicated, only `max_files` will be used in the analysis. This can
        be useful to test new functions on a subset of data.

    data_hdf_fname : str, optional
        Filename under which the HDF5 data summary is stored.
    """

    data_path: Union[Path, str]
    imgs_path: Union[Path, str]
    job_id: field(default="")
    max_files: field(default=0)
    data_hdf_fname: field(default="data.h5")

    def __post_init__(self):
        """Set up more attributes immediately after object initialisation."""
        self.data_path = Path(self.data_path)
        self.imgs_path = Path(self.imgs_path)
        try:
            self.imgs_path.mkdir(mode=0o760, parents=True, exist_ok=False)
            print(f"Creating path {self.imgs_path}")
        except FileExistsError:
            print(f"Path {self.imgs_path} already exists.")

        self.job_id = str(self.job_id)
        self.data_hdf_fname = self.imgs_path / self.data_hdf_fname
        self.sqlite_files = self.get_files()
        self.data = []

    def get_files(self):
        """Get list of all filenames to be used in the analysis."""
        sqlite_files = []
        for dirpath, _, filenames in os.walk(self.data_path):
            for fname in filenames:
                if fname.endswith(".sqlite") and self.job_id in fname:
                    sqlite_files.append(os.path.join(dirpath, fname))

        if self.max_files:
            sqlite_files = sqlite_files[: self.max_files]
        return sqlite_files

    @abstractmethod
    def append_data(self, data_dict: defaultdict(list), analyser: SqliteAnalyser):
        """Define which data should be gathered from each .sqlite file.

        This function does not need to have a return value.

        Parameters
        ----------
        data_dict : defaultdict(list)
            Append data to this defaultdict.

        analyser : SqliteAnalyser
            This object can be used to extract data from the .sqlite file.

        Example
        -------
        def append_data(self, data_dict):
            data_dict['TransferredMaterial'].append(analser.material_transfers(1, 2)
        """
        pass

    def get_data(self, append_data, force_update=False, store_data=True):
        """Extract data from sqlite files using MPI.

        Parameters
        ----------
        force_update : bool, optional
            If True, always extract data from sqlite files.
            If False (default), only do so in case no presaved data file
            (data.h5) is available.

        store_data : bool, optional
            Store data as .h5 file.
        """

        def print_mpi(msg, **kwargs):
            """Helper function to get consistent MPI output."""
            print(
                f"Rank {rank:2}, "
                f"{strftime('%y/%m/%d %H:%M:%S', localtime())}   " + msg,
                **kwargs,
            )

        if os.path.isfile(self.data_hdf_fname) and not force_update:
            self.data = pd.read_hdf(self.data_hdf_fname, key="df")
            print(f"Read in data from {self.data_hdf_fname}")
            return

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            # Maybe not the most elegant solution to distribute tasks but it works
            # and distributes them evenly.
            print_mpi(f"Analysing {len(self.sqlite_files)} in total.")
            n_per_task = len(self.sqlite_files) // size
            files_per_task = [
                self.sqlite_files[i * n_per_task : (i + 1) * n_per_task]
                for i in range(size)
            ]
            i = 0
            while i < len(self.sqlite_files) % size:
                files_per_task[i % size].append(
                    self.sqlite_files[size * n_per_task + i]
                )
                i += 1
        else:
            files_per_task = None

        files_per_task = comm.scatter(files_per_task, root=0)
        print_mpi(f"Received list of {len(files_per_task)} files.")

        # Keys: name of extracted data, values: lists with data
        data_dict = defaultdict(list)
        for i, fname in enumerate(files_per_task):
            if i % 20 == 0:
                print_mpi(f"{i:3}/{len(files_per_task)}")

            append_data(data_dict, SqliteAnalyser(fname, verbose=False))

        gatherdata = comm.gather(pd.DataFrame(data_dict), root=0)
        print_mpi("Gathered data")

        if rank != 0:
            print_mpi("Exiting function")
            sys.exit()
        print_mpi("Leaving parallelised section.")
        print_mpi("=============================\n")

        print_mpi("Concatenating dataframes")
        self.data = gatherdata[0]
        for gathered_data in gatherdata[1:]:
            self.data = pd.concat([self.data, gathered_data], axis=0, ignore_index=True)

        if store_data:
            self.data.to_hdf(self.data_hdf_fname, key="df", mode="w")
            print_mpi(f"Successfully stored data under {self.data_hdf_fname}.\n")

    def plot_1d_histogram(self, quantity, **hist_kwargs):
        """Generate seaborn 1D histogram."""
        _, ax = plt.subplots(constrained_layout=True)
        sns.histplot(data=self.data, x=quantity, **hist_kwargs, ax=ax)
        ax.legend(labels=[f"#entries={len(self.data[quantity])}"])
        ax.set_xlabel(quantity.replace("_", " "))

        plt.savefig(self.imgs_path / f"histogram_{quantity}.png")
        plt.close()

    def plot_all_1d_histograms(self, **hist_kwargs):
        """Generate 1D histograms for all quantities stored in the data.

        Parameters
        ----------
        hist_kwargs : kwargs
            Keyword arguments passed to seaborn.histplot.
        """
        for quantity in self.data.columns:
            self.plot_1d_histogram(quantity, **hist_kwargs)

    def plot_2d_scatterplots(self, x, y, marginals=False, **plot_kwargs):
        """Generate seaborn 2D histogram."""
        if marginals:
            g = sns.JointGrid(data=self.data, x=x, y=y)
            g.plot_joint(sns.scatterplot)
            g.plot_marginals(sns.histplot, kde=True)
            g.set_axis_labels(x.replace("_", " "), y.replace("_", " "))
            g.savefig(self.imgs_path / f"scatter_{x}_{y}.png")
            plt.close()

            return

        n_entries = len(self.data[x])
        _, ax = plt.subplots(constrained_layout=True)
        sns.scatterplot(data=self.data, x=x, y=y, **plot_kwargs, ax=ax)
        ax.legend(labels=[f"#entries={n_entries}"])
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        plt.savefig(self.imgs_path / f"scatter_{x}_{y}.png")
        plt.close()

    def pairplots(self, subset=None, fname=None):
        """Create a Seaborn pairplot.

        WARNING: This can be computationally expensive and can potentially
        generate very large plots.

        Parameters
        ----------
        subset : None or list of str, optional
            If None, use all data to generate the pairplot. If not None, this
            variable must contain a list of columns from self.data. Only these
            columns will be used to obtain the pairplot.
        """
        fname = "pairplot.png" if fname is None else fname
        data = self.data if subset is None else self.data[subset]

        pairplot_grid = sns.PairGrid(data, diag_sharey=False)
        pairplot_grid.map_upper(sns.kdeplot)
        pairplot_grid.map_diag(sns.histplot)
        pairplot_grid.map_lower(sns.scatterplot)
        plt.savefig(self.imgs_path / "pairplot.png")
        plt.close()
