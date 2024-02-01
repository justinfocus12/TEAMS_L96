# Systematic study of perturbation development. First run a "spine" simulation, like reanalysis, and then branch off of it at regular intervals with an ensemble of perturbations. We can investigate Lyapunov exponents/vectors and linear response this way. 

from abc import ABC,abstractmethod
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
svkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from numbers import Number
import xarray as xr
import dask
import netCDF4 as nc
from numpy.random import default_rng
import pickle
import sys
import os
from os.path import join, exists
from os import mkdir, makedirs
from collections import deque
import copy as copylib
from importlib import reload
from multiprocessing import pool as mppool

# Other modules in this folder
import ensemble
import utils


class PERTManager(ABC):
    def __init__(self, dirs, algo_params, init_pool):
        self.dirs = dirs.copy()
        for dirkey in list(self.dirs.keys()):
            os.makedirs(self.dirs[dirkey], exist_ok=False)
        self.algo_params = algo_params.copy() # time_units_per_chunk, 
        self.init_pool = init_pool # A queue of tuples of the form (file, time)

        self.scores_single = []
        self.scores_multiple = []
        #self.num_splits = [] # How many new children are birthed at each point along the trajectory
        self.max_scores = np.zeros(0)
        self.max_score_tidx = np.zeros(0, dtype=int)
        self.time_origins = np.zeros(0, dtype=int)
        self.times2split = np.zeros(0, dtype=int)

        self.rng_perturb_plant = default_rng(self.algo_params["seeddict"]["perturb_plant"]) 
        self.rng_perturb_branch = default_rng(self.algo_params["seeddict"]["perturb_branch"]) 

        return

    @abstractmethod
    def score_fun_multiple(self, score_multiple_ancestral):
        """
        Function to evaluate progress of a trajectory. Return an xarray.DataArray with a single coordinate, time, that matches the time of the hist_mem. 
        This score may include a rolling average or other time-dependence, which means the ancestor's score might be necessary for calculating it. 
        """
        pass

    @abstractmethod
    def score_fun_single(self, score_multiple):
        """
        This is what determines where to split the trajectories.
        """
        pass

    def link_model(self, ens):
        """
        """
        self.ens = ens
        self.acq_state_local = dict() # one key for each ancestor encoding what to do next in the local optimization/sampling
        self.initialize_acq_state_global() # Dependent upon the specific scheme, e.g., AMS or random 
        return

    def initialize_acq_state_global(self):
        asg = dict({
            "next_action": "plant",
            "previous_action": "",
            "ensembles_done": -1,
            })
        self.acq_state_global = asg
        return

    def update_acq_state_global(self):
        asg = self.acq_state_global.copy()

        # Decide next action
        if asg["ensembles_done"] == int(self.algo_params["time_horizon_spine"] / self.algo_params["hindcast_init_interval"]):
            asg["next_action"] = "terminate"
        else:
            asg["next_action"] = "branch"


        self.acq_state_global = asg
        return

    def initialize_acq_state_local(self):
        # refresh the random number generator for a new ensemble
        asl = dict({
            "members_done": 0,
            "member_list": np.zeros(0, dtype=int)
            })
        asg = self.acq_state_global
        i_ens = asg["ensembles_done"]
        t_split = i_ens*self.algo_params["hindcast_init_interval"] + self.algo_params["spinup_time"]
        print(f"{i_ens = }, {t_split = }")
        asl["t_split"] = t_split
        if self.ens.model_params["noise"]["type"] == "white":
            asl["next_pert_seq"] = xr.DataArray(
                    coords={"time": [t_split],},
                    dims=["time"],
                    data=self.rng_perturb_branch.integers(low=self.ens.model_params["seed_min"],high=self.ens.model_params["seed_max"],size=(1,)),
                    )
        self.acq_state_local[i_ens] = asl
        return

    def update_acq_state_local(self):
        i_ens = self.acq_state_global["ensembles_done"]
        asl = self.acq_state_local[i_ens]
        asl["members_done"] += 1
        asl["member_list"] = np.concatenate((asl["member_list"], [len(self.ens.mem_list)-1]))

        # Design the next perturbation sequence
        if self.ens.model_params["noise"]["type"] == "white":
            asl["next_pert_seq"] = xr.DataArray(
                    coords={"time": [asl["t_split"]],},
                    dims=["time",],
                    data=self.rng_perturb_branch.integers(low=self.ens.model_params["seed_min"],high=self.ens.model_params["seed_max"],size=(1,)),
                    )

        self.acq_state_local[i_ens] = asl
        return

    @abstractmethod
    def create_warmstart_from_file(self, init_file, init_time, pert_seq):
        pass
    @abstractmethod
    def create_warmstart_from_member(self, i_parent, init_time, pert_seq):
        pass

    def plant_tree(self, EnsMemClass):
        # Start the long simulation, to be branched off of later
        n_mem_init = 1 #self.algo_params["politics"]["batch_size"]
        memidx2run = len(self.ens.mem_list) + np.arange(n_mem_init)

        # Prepare the initial conditions
        for i_mem in memidx2run:
            init_file,init_time = self.init_pool.pop()
            print(f"About to feed in init file \n{init_file}\n and init_time {init_time}")
            warmstart_info = self.create_warmstart_from_file(init_file, init_time, self.algo_params["time_horizon_spine"]) 
            print(f"warmstart_info[init_cond] = \n{warmstart_info['init_cond']}")
            self.ens.initialize_new_member(EnsMemClass, warmstart_info.copy())
            self.time_origins = np.concatenate((self.time_origins, [init_time]))
            self.times2split = np.concatenate((self.times2split, [init_time]))
            
            chunks_per_mem = self.algo_params["chunks_per_mem"] * np.ones(len(memidx2run), dtype=int)

        # Run the simulation
        self.ens.run_batch(memidx2run, chunks_per_mem)

        for i_mem in memidx2run:
            mem = self.ens.mem_list[i_mem]
            # Replenish the pool of initial conditions
            # TODO build in some time gap to ensure independence in the case of nontrivial correlation between one timestep and the next...or simply ensure that the time horizon is always long enough. 
            self.init_pool.appendleft((mem.term_file_list[-1], mem.term_time_list[-1].item()))
            # Tally score and append the scoreboard
            hist_mem = mem.load_history_selfmade()
            score_mult = self.score_fun_multiple(hist_mem) 
            score_sing = self.score_fun_single(score_mult) # TODO build in history of the initialization, if possible
            self.scores_multiple.append(score_mult)
            self.scores_single.append(score_sing)
            scm = score_sing.max(dim="time").compute().item()
            self.max_scores = np.concatenate((self.max_scores, [scm]))
            self.max_score_tidx = np.concatenate((self.max_score_tidx, [score_sing.argmax(dim="time").item()]))

        print(f"done")
        return

    def branch(self, EnsMemClass):
        asg = self.acq_state_global
        asl = self.acq_state_local[asg["ensembles_done"]]
        parent = 0
        pert_seq = asl["next_pert_seq"]
        t_split = asl["t_split"]

        # Prepare the new trajectory
        child = len(self.ens.mem_list)
        warmstart_info = self.create_warmstart_from_member(parent, t_split, pert_seq, self.algo_params["time_horizon_hindcast"])
        print(f"Inside branch: {t_split = }")
        self.ens.initialize_new_member(EnsMemClass, warmstart_info, parent)

        # Run the new trajectory
        memidx2run = [child]
        chunks_per_mem = self.algo_params["chunks_per_mem"] * np.ones(1, dtype=int)
        self.ens.run_batch(memidx2run, chunks_per_mem)

        # Record the child's score
        score_parent_mult = self.scores_multiple[parent]
        time_origin = t_split
        hist_child = self.ens.mem_list[child].load_history_selfmade()
        score_child_mult = self.score_fun_multiple(hist_child, score_parent_mult)
        score_child_sing = self.score_fun_single(score_child_mult)
        self.scores_multiple.append(score_child_mult.sel(time=slice(time_origin,None))) 
        self.scores_single.append(score_child_sing.sel(time=slice(time_origin,None))) 
        new_max_score = self.scores_single[-1].max(dim="time").compute().item()
        self.max_scores = np.concatenate((self.max_scores, [new_max_score]))
        self.max_score_tidx = np.concatenate((self.max_score_tidx, [score_child_sing.argmax(dim="time").item()]))
        self.time_origins = np.concatenate((self.time_origins, [time_origin]))
        self.times2split = np.concatenate((self.times2split, [t_split]))

        return

    def take_next_step(self, EnsMemClass):
        if self.acq_state_global["next_action"] == "terminate":
            print(f"Terminating")
        else:
            if self.acq_state_global["next_action"] == "plant":
                self.plant_tree(EnsMemClass)
                self.acq_state_global["previous_action"] = "plant"

            elif self.acq_state_global["next_action"] == "branch":
                self.branch(EnsMemClass)
                self.acq_state_global["previous_action"] = "branch"

            # -------------- Determine the next action -------------------
            new_member = len(self.ens.mem_list)-1
            self.update_acq_state_global()
            if self.acq_state_global["previous_action"] == "branch":
                self.update_acq_state_local() 
            if self.acq_state_global["previous_action"] == "plant" or self.acq_state_local[self.acq_state_global["ensembles_done"]]["members_done"] == self.algo_params["hindcast_ensemble_size"]:
                self.acq_state_global["ensembles_done"] += 1
                print(f"initializing asl")
                self.initialize_acq_state_local()
                print(f"{self.acq_state_local[self.acq_state_global['ensembles_done']] = }")
            self.save_state()
        self.print_status()
        return

    def print_status(self):
        print(f" ! ! ! ! ! -------- Status report ------- ! ! ! ! ! ")
        print(f"\tNumber of members = {len(self.ens.mem_list)}")
        print(f"\tSpine: max score = {self.max_scores[0]}")
        asg = self.acq_state_global
        for i_ens in range(asg["ensembles_done"]):
            asl = self.acq_state_local[i_ens]
            print(f"\tEnsemble {i_ens}: memlist = {asl['member_list'][0]}-{asl['member_list'][-1]},  min = {np.min(self.max_scores[asl['member_list']])}, max = {np.max(self.max_scores[asl['member_list']])}")
        print(f" ! ! ! ! ! -------- End status report ------- ! ! ! ! ! ")
        return
        
    def save_state(self):
        filename = join(self.dirs["metadata"], f"manager")
        if exists(filename): 
            os.rename(filename, join(self.dirs["metadata"], "backup_manager"))
        pickle.dump(self, open(join(self.dirs["metadata"], f"manager"), "wb"))
        return

    @classmethod
    def quantify_divergence_rates(cls, EnsClass, manager, savedir):
        # Calculate finite-time or finite-size Lyapunov exponents, and aggregate the statistics together. Also compute the time-to-quarter-RMSE
        num_thresh = 16
        rho = 2 # Growth ratio from one threshold to the next
        asg = manager.acq_state_global
        tau = xr.DataArray(coords={"ensemble": np.arange(asg["ensembles_done"]), "member": np.arange(manager.algo_params["hindcast_ensemble_size"]), "n": np.arange(num_thresh+1)}, dims=["ensemble","member","n"], data=np.nan)
        delta = xr.DataArray(coords={"ensemble": np.arange(asg["ensembles_done"]), "member": np.arange(manager.algo_params["hindcast_ensemble_size"]), "n": np.arange(num_thresh+1)}, dims=["ensemble","member","n"], data=np.nan)
        # Calculate RMSD over the attractor (trace of covariance matrix)
        hist_anc = manager.ens.mem_list[0].load_history_selfmade()
        print(f"{np.unique(manager.times2split) = }")
        print(f"{hist_anc.time = }")
        dns_snapshots = xr.concat([hist_anc.sel(time=manager.acq_state_local[i_ens]["t_split"]) for i_ens in range(asg["ensembles_done"])], dim="time").load()
        rmsd = 0.0
        npairs = 0
        for i0 in range(dns_snapshots.time.size):
            for i1 in range(i0+1,dns_snapshots.time.size):
                rmsd += EnsClass.norm(dns_snapshots.isel(time=i0) - dns_snapshots.isel(time=i1)).item()
                npairs += 1
        rmsd /= npairs
        print(f"{rmsd = }")
            
        rmsdft = xr.DataArray(
                coords={"ensemble": np.arange(asg["ensembles_done"]), "rmsdfrac": np.array([
                    1/32,1/16,1/8,1/4,3/8,1/2,1-3/8,1-1/4,1-1/8,1-1/16,1-1/32
                    ]),},
                dims=["ensemble","rmsdfrac"],
                data=np.nan,
                attrs={"rmsd": rmsd},
                )
        # Define thresholds
        for i_ens in range(asg["ensembles_done"]):
            asl = manager.acq_state_local[i_ens]
            timesel = slice(asl["t_split"], asl["t_split"] + manager.algo_params["time_horizon_hindcast"])
            hist_anc_ens = hist_anc.sel(time=timesel).load()
            hist_ens = []
            for i_mem in asl["member_list"]:
                hist_ens.append(manager.ens.mem_list[i_mem].load_history_selfmade().load())
            hist_ens = xr.concat(hist_ens, dim="member")
            rse = EnsClass.norm(hist_ens - hist_anc_ens)
            rmse = rse.mean("member")
            for frac in rmsdft["rmsdfrac"].to_numpy():
                exceedances = np.where(rmse > frac*rmsd)[0]
                if len(exceedances) > 0:
                    rmsdft.loc[dict(ensemble=i_ens,rmsdfrac=frac)] = exceedances[0]

        rmsdft.to_netcdf(join(savedir,"rmsdft.nc"))
        


        return

    @classmethod
    def plot_member_spaghetti(cls, EnsClass, manager, savedir, paramdisp, state_label="proto-score", score_label="score"):
        labelsize = 24
        titlesize = 28
        ticksize = 22
        # Plot all the ensembles atop each other
        tu = manager.ens.model_params["time_unit"]
        print(f"{tu = }")
        ttrans = lambda da: da.assign_coords({"time": (da["time"]-da["time"].isel(time=0).item())*tu})
        timescale_fwd = lambda t: t*tu
        timescale_inv = lambda t: t/tu
        asg = manager.acq_state_global
        rmsdft = xr.open_dataarray(join(savedir,"rmsdft.nc"), decode_times=False)
        rmsdft_ensmean = rmsdft.mean(dim="ensemble")
        rmsd = rmsdft.attrs["rmsd"]
        for i_ens in range(asg["ensembles_done"]):
            asl = manager.acq_state_local[i_ens]
            timesel = slice(asl["t_split"], asl["t_split"]+manager.algo_params["time_horizon_hindcast"]+1)
            hist_anc = manager.ens.mem_list[0].load_history_selfmade().sel(time=timesel).load()
            fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,10),sharex=True)

            # Row 0: the proto-score
            ax = axes[0]
            print(f"{i_ens = }, {asl['member_list'] = }, {manager.time_origins[asl['member_list']] = }")
            for i_mem in asl["member_list"]:
                xr.plot.plot(ttrans(manager.scores_multiple[i_mem]), x="time", color="tomato", linestyle="-", linewidth=1, ax=ax)
                print(f'data just plotted = \n{ttrans(manager.scores_multiple[i_mem])}')
            ax.axvline(0, color="tomato")
            #ax.axvline(manager.time_origins[asl["member_list"][0]]*tu, color="red")
            xr.plot.plot(ttrans(manager.scores_multiple[0].sel(time=timesel)), x="time", color="black", linestyle="--", linewidth=3, ax=ax)
            ymin = np.min(manager.scores_multiple[0])
            ymax = 1.5*np.max(manager.scores_multiple[0]) - 0.5*ymin
            #ax.set_ylim([ymin,ymax])
            ax.set_title(paramdisp, fontsize=titlesize)
            ax.set_xlabel("", fontsize=labelsize)
            ax.set_ylabel(state_label, fontsize=labelsize)
            ax.tick_params(axis='both',which='both',labelleft=True,labelbottom=True,labelsize=ticksize)
            
            # Row 1: Euclidean distance between state vectors (1), and as fraction of saturation (2)
            ax = axes[1]
            print(f"{i_ens = }, {asl['member_list'] = }, {manager.time_origins[asl['member_list']] = }")
            for i_mem in asl["member_list"]:
                hist_mem = manager.ens.mem_list[i_mem].load_history_selfmade().load()
                dist = EnsClass.norm(hist_mem - hist_anc)
                xr.plot.plot(ttrans(dist)/rmsd, x="time", color="tomato", ax=ax)
                print(f'{dist = }, {rmsd = }')

            ax.plot((rmsdft_ensmean.to_numpy())*tu, rmsdft["rmsdfrac"].to_numpy(), color="black", marker="o", linewidth=2)
            ax.axhline(1.0, color="gray", linewidth=3)
            ax.set_ylabel("Distance/RMSD", fontsize=labelsize)
            ax.set_title("")
            ax.set_xlabel(r"Time since initializatin $t_0=%g$"%(dist.time[0].item()*tu), fontsize=labelsize)
            ax.tick_params(axis='both',which='both',labelleft=True,labelbottom=True,labelsize=ticksize)
            # Trim all horizontal axes 
            duration = (timesel.stop - timesel.start)*tu
            print(f'{duration = }')
            for ax in axes:
                print(f'{ax.get_xlim() = }')
                ax.set_xlim(np.array([0,0.6*duration]))
                print(f'{ax.get_xlim() = }')
            print(f'filename = {join(savedir,f"scores_ens{i_ens}")}')
            fig.savefig(join(savedir,f"scores_ens{i_ens}"), **svkwargs)
            plt.close(fig)

        
        return
