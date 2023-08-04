#!/usr/bin/env python3

"""Collection of classes and methods to analyse Cyclus .sqlite output files."""

import os
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


class SqliteAnalyser:
    """Class to obtain data from a single Cyclus .sqlite output files.

    If the functions provided here are not sufficient, create a subclass
    inheriting from SqliteAnalyser. Or feel free to submit a pull request and
    add the function to the repository.
    """

    def __init__(self, fname, verbose=True):
        """Initialise Analyser object.

        Parameters
        ----------
        fname : Path
            Path to .sqlite file

        verbose : bool, optional
            If true, increase verbosity of output.
        """
        self.fname = fname
        if not os.path.isfile(self.fname):
            msg = f"File {os.path.abspath(self.fname)} does not exist!"
            raise FileNotFoundError(msg)

        self.verbose = verbose
        if self.verbose:
            print(f"Opened connection to file {self.fname}.")

        self.connection = sqlite3.connect(self.fname)
        self.cursor = self.connection.cursor()
        self.duration = self.cursor.execute("SELECT Duration FROM Info").fetchone()[0]

        # Dataframe with multiple columns including 'Name', 'Spec', and index 'Id'.
        self.agents = self.get_agents()

    def __del__(self):
        """Destructor closes the connection to the file."""
        try:
            self.connection.close()
            if self.verbose:
                print(f"Closed connection to file {self.fname}.")
        except AttributeError as e:
            raise RuntimeError(f"Error while closing file {self.fname}") from e

    def get_agents(self):
        """Get all agents that took part in the simulation.

        Returns
        -------
        pd.DataFrame with index 'Id' and columns 'Name', 'Spec', 'EnterTime', 'Lifetime'.
        """
        columns = ["Id", "Name", "Spec", "EnterTime", "Lifetime"]
        query = self.cursor.execute(
            f"SELECT {', '.join(columns)} FROM AgentEntry"
        ).fetchall()
        agent_df = pd.DataFrame(data=query, columns=columns)
        # Get rid of the library, e.g., change ":agent:NullRegion" to "NullRegion".
        agent_df["Spec"] = agent_df["Spec"].map(lambda x: x.split(":")[-1])
        agent_df.set_index("Id", inplace=True)

        return agent_df

    def agent_id_or_name(self, agent_id_or_name):
        """Helper function to convert agent names into agent IDs."""
        if isinstance(agent_id_or_name, str):
            try:
                return self.agents.query(f"Prototype == '{agent_id_or_name}'").index[0]
            except KeyError as e:
                msg = f"Invalid agent name! Valid names are {self.agents['Name']}."
                raise KeyError(msg) from e
        elif isinstance(agent_id_or_name, int):
            return agent_id_or_name
        else:
            raise ValueError("`agent_id_or_name` must be agent name (str) or id (int)!")

    def material_transfers(self, sender_id_or_name, receiver_id_or_name, sum_=True):
        """Get all material transfers between two facilities.

        Parameters
        ----------
        sender_id_or_name, receiver_id_or_name : int or str
            Agent IDs or agent names of sender and receiver, respectively. Use
            '-1' as placeholder for 'all facilities'.

        sum_ : bool, optional
            If true, sum over all timesteps.

        Returns
        -------
        transfer_array : np.array of shape (number of transfers, 2)
            The first element of each entry is the timestep of the transfer,
            the second is its mass. If sum_ is True, then the second element
            is the total mass (over all timesteps) and the time is set to -1.
        """
        sender_id = self.agent_id_or_name(sender_id_or_name)
        receiver_id = self.agent_id_or_name(receiver_id_or_name)

        sender_cond = "" if sender_id == -1 else "SenderId = :sender_id "
        recv_cond = "" if receiver_id == -1 else "ReceiverId = :receiver_id "

        if sender_cond and recv_cond:
            sqlite_condition = f"WHERE ({sender_cond} AND {recv_cond})"
        else:
            sqlite_condition = f"WHERE ({sender_cond}{recv_cond})"

        sender_receiver = {"sender_id": sender_id, "receiver_id": receiver_id}
        transfer_times = self.cursor.execute(
            f"SELECT Time FROM Transactions {sqlite_condition}", sender_receiver
        ).fetchall()
        transfer_masses = self.cursor.execute(
            "SELECT Quantity FROM Resources WHERE ResourceId IN "
            f"(SELECT ResourceId FROM Transactions {sqlite_condition});",
            sender_receiver,
        ).fetchall()

        transfer_array = np.array(
            [[time[0], mass[0]] for time, mass in zip(transfer_times, transfer_masses)]
        )

        if sum_:
            return np.array([-1, transfer_array[:, 1].sum()])

        return transfer_array

    def enrichment_feeds(self, agent_id_or_name):
        """Get the amount of each enrichment feed that got used."""
        agent_id = self.agent_id_or_name(agent_id_or_name)
        query = self.cursor.execute(
            "SELECT Value, Units FROM TimeSeriesEnrichmentFeed WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchall()
        feeds = defaultdict(float)
        for value, feed in query:
            feeds[feed] += value
        return feeds

    def swu_available(self, agent_id_or_name, sum_=True):
        """Get the SWU available to one enrichment facility.

        Parameters
        ----------
        agent_id_or_name : str or int
            Agent ID or agent name

        sum_ : bool, optional
            If True, only yield the total available SWU (summed over all
            timesteps).

        Returns
        -------
        np.array of shape (number of timesteps, 2) if `sum_` is False, else of
        shape (-2)
        """
        agent_id = self.agent_id_or_name(agent_id_or_name)
        data = self.cursor.execute(
            "SELECT swu_capacity_times, swu_capacity_vals FROM "
            "AgentState_flexicamore_FlexibleEnrichmentInfo WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchone()  # (Boost vector with times, Boost vector with SWUs)

        # Convert Boost vectors (in XML) into Python lists.
        swu_times = []
        swu_vals = []
        for list_, cyclus_data in zip(
            (swu_times, swu_vals), [BeautifulSoup(d, "xml") for d in data]
        ):
            for item_ in cyclus_data.find_all("item"):
                list_.append(float(item_.get_text()))

        # Fill with timesteps where SWU was not changed.
        complete_list = []
        previous_time = swu_times[0]
        previous_val = swu_vals[0]
        for time, val in zip(swu_times, swu_vals):
            for t in range(int(previous_time), int(time)):
                complete_list += [[t, previous_val]]
            previous_time = time
            previous_val = val

        # Convert to array and transform timesteps since deployment to timesteps since
        # start of the simulation.
        swu_available = np.array(complete_list)
        swu_available[:, 0] += self.agents.loc[agent_id, "EnterTime"]

        if sum_:
            return np.array([-1, swu_available[:, 1].sum()])
        return swu_available

    def swu_used(self, agent_id_or_name, sum_=True):
        """Get the SWU used by one enrichment facility.

        Parameters
        ----------
        agent_id_or_name : str or int
            Agent ID or agent name

        sum_ : bool, optional
            If True, only yield the total SWU used (summed over all timesteps).

        Returns
        -------
        np.array of shape (number of timesteps, 2) if `sum_` is False, else of
        shape (-2)
        """
        agent_id = self.agent_id_or_name(agent_id_or_name)
        query = self.cursor.execute(
            "SELECT Time, Value FROM TimeSeriesEnrichmentSWU WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchall()
        rval = np.array(query, dtype=float)
        if sum_:
            return np.array([-1, rval[:, 1].sum()])

        return rval

    def capacity_factor_planned(self, agent_id_or_name):
        """Get the planned capacity factor of one reactor.

        Note that this value corresponds to the capacity factor *as indicated
        in Cyclus' input file. Thus, the actual capacity factor (online time /
        total time) may be smaller than this value, e.g., in case of missing
        fresh fuel.

        Parameters
        ----------
        agent_id_or_name : str or int
            Agent ID or agent name

        Returns
        -------
        dict with keys cycle_time, refuelling_time, capacity_factor
        """
        agent_id = self.agent_id_or_name(agent_id_or_name)
        if self.agents.loc[agent_id, "Spec"] != "Reactor":
            msg = f"Agent ID {agent_id} does not correspond to a 'Reactor' facility"
            raise ValueError(msg)

        cycle_time, refuelling_time = self.cursor.execute(
            "SELECT cycle_time, refuel_time FROM AgentState_cycamore_ReactorInfo "
            "WHERE AgentId = :agent_id",
            {"agent_id": agent_id},
        ).fetchone()
        return {
            "cycle_time": cycle_time,
            "refuelling_time": refuelling_time,
            "capacity_factor_planned": cycle_time / (cycle_time + refuelling_time),
        }

    def reactor_operations(self, agent_id_or_name):
        """Calculate reactor stats such as the effective capacity factor.

        Parameters
        ----------
        agent_id_or_name : str or int
            Agent ID or agent name

        Returns
        -------
        dict with keys:
            'n_start',
            'n_end',
            'cycle_time',
            'refuelling_time',
            'capacity_factor_planned',
            'capacity_factor_used',
            'in_sim_time',
            'total_cf_time',

            Note that 'total_cf_time' is the total time considered in the
            calculation of the capacity factor.
        """
        agent_id = self.agent_id_or_name(agent_id_or_name)
        data = self.capacity_factor_planned(agent_id)
        lifetime = self.agents.loc[agent_id, "Lifetime"]
        in_sim_time = (
            lifetime
            if lifetime != -1
            else self.duration - self.agents.loc[agent_id, "EnterTime"]
        )
        data["in_sim_time"] = in_sim_time

        # Keywords used in 'ReactorEvents' table.
        cycle_start = "CYCLE_START"
        cycle_end = "CYCLE_END"

        reactor_events = self.cursor.execute(
            "SELECT Time, Event FROM ReactorEvents WHERE (AgentId = :agent_id)"
            " AND (Event = :cycle_start or Event = :cycle_end)",
            {"agent_id": agent_id, "cycle_start": cycle_start, "cycle_end": cycle_end},
        ).fetchall()
        data["n_start"] = sum(event == cycle_start for _, event in reactor_events)
        data["n_end"] = sum(event == cycle_end for _, event in reactor_events)

        # We cannot calculate the capacity factor if there are not at least
        # three events (i.e., cycle start, cycle end and another cycle start).
        if len(reactor_events) < 3:
            msg = "Need at least 3 events to be able to calculate the capacity factor."
            raise RuntimeError(msg)

        # We only consider complete cycles including refueling period.
        if reactor_events[-1][1] == cycle_start:
            online_time = data["n_end"] * data["cycle_time"]
            total_time = reactor_events[-1][0] - reactor_events[0][0]
        else:  # Last event = cycle_end
            time_to_sim_end = self.duration - reactor_events[-1][0]
            if time_to_sim_end > data["refuelling_time"]:
                # New cycle could have been started but did not --> This needs to be
                # taken into account in the capacity factor.
                online_time = data["n_end"] * data["cycle_time"]
                total_time = self.duration - reactor_events[0][0]
            else:
                # New cycle could not have been started before end of simulation
                # --> Do not take this cycle into account.
                online_time = (data["n_end"] - 1) * data["cycle_time"]
                total_time = reactor_events[-2][0] - reactor_events[0][0]

        data["capacity_factor_used"] = online_time / total_time
        data["total_cf_time"] = total_time

        return data

    def all_reactor_operations(self, spec="Reactor"):
        """Get an overview over all reactor operations for all reactors.

        Returns
        -------
        dict: (str, int) -> dict: str -> float
            Keys of the 'outer' dict are 'total' and all reactor agent IDs.
            The 'inner' dict contains the following keys:
                'n_start',
                'n_end',
                'cycle_time',
                'refuelling_time',
                'capacity_factor_planned',
                'capacity_factor_used',
                'in_sim_time',
                'total_cf_time',
            except for the 'total' dict which contains only a subset of keys:
                'n_start',
                'n_end',
                'in_sim_time',
                'capacity_factor_planned',
                'capacity_factor_used',
        """
        agent_ids = self.agents.query(f"Spec == '{spec}'").index.values

        all_reactor_ops = {"total": defaultdict(float)}
        for agent_id in agent_ids:
            reactor_ops = self.reactor_operations(agent_id)
            all_reactor_ops[agent_id] = reactor_ops
            for k in ("n_start", "n_end", "in_sim_time"):
                all_reactor_ops["total"][k] += reactor_ops[k]

        all_reactor_ops["total"]["capacity_factor_planned"] = (
            sum(
                v["in_sim_time"] * v["capacity_factor_planned"]
                for k, v in all_reactor_ops.items()
                if k != "total"
            )
            / all_reactor_ops["total"]["in_sim_time"]
        )
        all_reactor_ops["total"]["capacity_factor_used"] = sum(
            v["total_cf_time"] * v["capacity_factor_used"]
            for k, v in all_reactor_ops.items()
            if k != "total"
        ) / sum(v["total_cf_time"] for k, v in all_reactor_ops.items() if k != "total")

        return all_reactor_ops
