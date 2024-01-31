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
from scipy.stats import norm as spnorm, beta as spbeta
from scipy.special import logsumexp,softmax
from sklearn.neighbors import KernelDensity
from sklearn.gaussian_process import GaussianProcessRegressor
import sys
import os
from os.path import join, exists
from os import mkdir, makedirs
from collections import deque
import shutil
import copy as copylib
from importlib import reload
from multiprocessing import pool as mppool

# Other modules in this folder
import ensemble
import utils


class PRTManager(ABC):
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
        if self.ens.model_params["noise"]["type"] == "jump":
            ndim = self.ens.model_params["noise_dim"]
            random_vector = asl["rng"].normal(size=(1,ndim))
            random_vector /= np.sqrt(np.sum(random_vector**2))
            asl["next_pert_seq"] = xr.DataArray(
                    coords={"time": [t_split], "component": np.arange(ndim),},
                    dims=["time","component"],
                    data=random_vector)
        elif self.ens.model_params["noise"]["type"] == "white":
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
        if self.ens.model_params["noise"]["type"] == "jump":
            ndim = self.ens.model_params["noise_dim"]
            random_vector = asl["rng"].normal(size=(1,ndim))
            random_vector /= np.sqrt(np.sum(random_vector**2))
            asl["next_pert_seq"] = xr.DataArray(
                    coords={"time": [asl["t_split"]], "component": np.arange(ndim),},
                    dims=["time","component"],
                    data=random_vector)
        elif self.ens.model_params["noise"]["type"] == "white":
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
            if False:
                tidx0 = 10*manager.ens.model_params["sims_per_unit"]
                dist_ratio = rse / rse.isel(time=tidx0,drop=True)
                for i_mem in range(len(asl["member_list"])):
                    t0 = asl["t_split"] + tidx0
                    delta[dict(ensemble=i_ens,member=i_mem,n=0)] = rse.isel(time=tidx0,member=i_mem)
                    for n in np.arange(num_thresh):
                        idx = np.where(dist_ratio.isel(member=i_mem).to_numpy() > rho**(n+1))[0]
                        if len(idx) > 0:
                            t1 = dist_ratio["time"][idx[0]].item()
                            tau[dict(ensemble=i_ens,member=i_mem, n=n)] = t1 - t0
                            delta[dict(ensemble=i_ens,member=i_mem, n=n)] = rse.isel(member=i_mem,time=idx[0])
                            t0 = t1

        #tau.to_netcdf(join(savedir,"tau.nc"))
        #delta.to_netcdf(join(savedir,"delta.nc"))
        rmsdft.to_netcdf(join(savedir,"rmsdft.nc"))
        #print(f"{tau = }")
        #print(f"{delta = }")
        print(f"{rmsdft = }")
        
        if False:
            # Compute the FSLE 
            fsle = xr.DataArray(
                    coords={"ensemble": delta["ensemble"], "delta": delta.mean(dim=["ensemble","member"]).to_numpy()},
                    dims=["ensemble","delta"],
                    data=(np.log(rho)/tau.where(tau>0, np.nan)).mean(dim="member"))
            fsle.attrs = {"rho": rho, "t0": tidx0}
            fsle.to_netcdf(join(savedir,"fsle.nc"))

            # Plot doubling time vs. amplitude 
            for i_ens in range(asg["ensembles_done"]):
                asl = manager.acq_state_local[i_ens]
                fig,axes = plt.subplots(nrows=2,figsize=(8,8),sharex=True)
                ax = axes[0]
                ax.set_title(r"$\rho=%.2f$"%(rho))
                for i_mem in range(len(asl["member_list"])):
                    ax.plot(delta.isel(ensemble=i_ens,member=i_mem), tau.isel(ensemble=i_ens,member=i_mem), marker=".")
                ax.set_ylabel(r"$\rho$-folding time $\tau(\delta)$")
                ax = axes[1]
                for i_mem in range(len(asl["member_list"])):
                    ax.plot(delta.isel(ensemble=i_ens,member=i_mem,n=slice(1,None)), np.log(rho)/tau.isel(ensemble=i_ens,member=i_mem,n=slice(1,None)), marker=".")
                xr.plot.plot(fsle.isel(ensemble=i_ens), x="delta", color="black", linewidth=3, linestyle="--")
                ax.set_xscale("log")
                ax.set_xlabel(r"$\delta$")
                ax.set_ylabel(r"FSLE $\log(\rho)/\tau(\delta)$")
                fig.savefig(join(savedir,f"fsle_ensemble{i_ens}"), **svkwargs)
                plt.close(fig)


            # Detect the change-point from exponential to linear growth of perturbation error
            # Take saturation time as the sum of tau(delta) up to the last delta that actually does double
            # Saturation time 
            sat_time = xr.DataArray(
                    coords={"ensemble": delta["ensemble"]},
                    dims=["ensemble"],
                    data=np.nan
                    )
            for i_ens in range(asg["ensembles_done"]):
                asl = manager.acq_state_local[i_ens]
                sat_times_ens = []
                for i_mem in range(len(asl["member_list"])):
                    tau_finite_idx = np.where(np.isfinite(tau.isel(ensemble=i_ens,member=i_mem)))[0]
                    if len(tau_finite_idx) > 0:
                        new_sat_time = fsle.attrs["t0"] + tau.isel(ensemble=i_ens,member=i_mem,n=slice(None,tau_finite_idx[-1])).sum(dim="n")
                    else:
                        new_sat_time = np.nan
                    sat_times_ens.append(new_sat_time)
                sat_time[dict(ensemble=i_ens)] = np.mean(sat_times_ens)
            sat_time.to_netcdf(join(savedir,"sat_time.nc"))
            print(f"{sat_time = }")


        return

    @classmethod
    def plot_pdf_change(cls, EnsClass, manager, savedir, ens_dss, paramdisp):
        # Plot the pdf of block maxima (for various time horizons) with and without the branched trajectories
        score_dns = manager.scores_single[0]
        score_dns_long = manager.score_fun_single(manager.score_fun_multiple(ens_dss.load_member_ancestry(0))).compute()
        Thorz = manager.algo_params["time_horizon_hindcast"]
        T = Thorz - manager.algo_params["score"]["twait"] - (manager.algo_params["score"]["tavg"] - 1)
        tu = manager.ens.model_params["time_unit"]
        n_blocks_dns = int(round(score_dns["time"].size/T))
        n_blocks_dns_long = int(round(score_dns_long["time"].size/T))
        lsf_interp = np.linspace(-np.log(n_blocks_dns_long), np.log(0.5), 30)[::-1]
        rlev = dict()
        rlsf = dict()
        gevpar = dict()
        hist = dict()
        rlev["dns"],rlsf["dns"],dns_block_maxima,gevpar["dns"],min_level = utils.estimate_return_level_mbm(score_dns, T, boot_sample_size=n_blocks_dns,n_boot=250, min_quantile=0.0,lsf_interp=lsf_interp)
        rlev["dns_long"],rlsf["dns_long"],dns_block_maxima_long,gevpar["dns_long"],min_level = utils.estimate_return_level_mbm(score_dns_long, T, boot_sample_size=n_blocks_dns_long,n_boot=250, min_quantile=0.0,lsf_interp=lsf_interp)
        # Shorter DNS estimate overlapping with PRT
        prt_fintime = manager.times2split[-1]+Thorz
        print(f"{prt_fintime =  }")
        print(f"{manager.scores_single[0]['time'][-1].item() = }")
        n_blocks_dns_overlap = int(round(prt_fintime/T))
        rlev["dns_overlap"],rlsf["dns_overlap"],dns_overlap_block_maxima,gevpar["dns_overlap"],_ = utils.estimate_return_level_mbm(score_dns.sel(time=slice(None,prt_fintime+1)), T, boot_sample_size=n_blocks_dns_overlap,n_boot=250, min_quantile=0.0,lsf_interp=lsf_interp)
        rlev["prt"],rlsf["prt"],gevpar["prt"] = utils.estimate_return_statistics_one_ensemble(manager.max_scores[1:],np.zeros(len(manager.ens.mem_list)-1),lsf_interp=lsf_interp)

        def lsf2rt_vals(ppp):
            rrr = -T*tu/np.log1p(-np.exp(ppp))
            return rrr
        rt = dict({key: lsf2rt_vals(rlsf[key]) for key in ["dns","prt","dns_overlap","dns_long"]})

        # Also compute histograms, no block maxima, as simpler test
        ymin = np.min(manager.scores_single[0])
        ymax = 1.5*np.max(manager.scores_single[0]) - 0.5*ymin
        bin_edges = np.linspace(ymin, ymax, 30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        hist["dns"],_ = np.histogram(score_dns.to_numpy(), bins=bin_edges, density=True)
        hist["dns_overlap"],_ = np.histogram(score_dns.sel(time=slice(None,prt_fintime+1)).to_numpy(), bins=bin_edges, density=True)
        hist["dns_long"],_ = np.histogram(score_dns_long.to_numpy(), bins=bin_edges, density=True)
        prt_scores_all = np.array([manager.scores_single[i_mem].to_numpy() for i_mem in range(1,len(manager.ens.mem_list))])
        print(f"{prt_scores_all.shape = }")
        hist["prt"] = dict()
        Thorz_frac_list = np.array([1/8,1/4,1/2,1])
        for Thorz_frac in Thorz_frac_list:
            hist["prt"][Thorz_frac],_ = np.histogram(prt_scores_all[:,:int(Thorz_frac*Thorz)], bins=bin_edges, density=True)

        # Plot both the return periods and the histograms
        fig,axes = plt.subplots(ncols=2,figsize=(10,5),sharey=True)
        ax = axes[0]
        hdns, = xr.plot.plot(rt["dns_long"].sel(confint=0,side="lo",est="empirical"), y="lev", color="black", ax=ax, label="DNS long")
        ax.fill_betweenx(rt["dns_long"]["lev"],rt["dns"].sel(confint=0.95,est="empirical",side="lo"),rt["dns_long"].sel(confint=0.95,est="empirical",side="hi"),color="gray",zorder=-1,alpha=0.25)
        #hdns_overlap, = xr.plot.plot(rt["dns_overlap"].sel(confint=0,side="lo",est="empirical"), y="lev", color="dodgerblue", ax=ax, label="DNS overlapping")
        hprt, = xr.plot.plot(rt["prt"].sel(est="empirical"), y="lev", color="red", ax=ax, label="PERT")
        print(f"{rt['dns_overlap'] = }")
        ax.set_xscale("log")
        ax.legend(handles=[hdns,hprt])
        ax.set_xlabel("Return time")
        ax.set_ylabel("Return level")
        ax.set_title("")
        ymin = np.min(manager.scores_single[0])
        ymax = 1.5*np.max(manager.scores_single[0]) - 0.5*ymin
        ax.set_ylim([ymin,ymax])

        ax = axes[1]
        handles = []
        h, = ax.plot(hist["dns"], bin_centers, color="dodgerblue", label="DNS")
        handles.append(h)
        h, = ax.plot(hist["dns_overlap"], bin_centers, color="dodgerblue", linestyle="--", label="DNS overlapping")
        handles.append(h)
        h, = ax.plot(hist["dns_long"], bin_centers, color="black", linestyle="--", label="DNS long")
        handles.append(h)
        for Thorz_frac in Thorz_frac_list:
            h, = ax.plot(hist["prt"][Thorz_frac], bin_centers, label=r"PERT ($T=%.2f$)"%(Thorz*Thorz_frac*tu), color=plt.cm.Reds(Thorz_frac))
            handles.append(h)
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1,1))
        ax.set_xscale("log")
        ax.set_xlabel("Prob. dens.")


        fig.suptitle(paramdisp)
        fig.subplots_adjust(top=0.85)
        fig.savefig(join(savedir,"returnplot.png"),**svkwargs)
        plt.close(fig)
        
        
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
        if False:
            delta = xr.open_dataarray(join(savedir,"delta.nc"), decode_times=False)
            tau = xr.open_dataarray(join(savedir,"tau.nc"), decode_times=False)
            fsle = xr.open_dataarray(join(savedir,"fsle.nc"), decode_times=False)
            sat_time = xr.open_dataarray(join(savedir,"sat_time.nc"), decode_times=False)
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
            if False:
                # Row 3: score function
                ax = axes[3]
                print(f"{i_ens = }, {asl['member_list'] = }, {manager.time_origins[asl['member_list']] = }")
                for i_mem in asl["member_list"]:
                    xr.plot.plot(ttrans(manager.scores_single[i_mem]), x="time", color="red", linestyle="-", linewidth=1, ax=ax)
                ax.axvline(manager.time_origins[asl["member_list"][0]]*tu, color="red")
                xr.plot.plot(ttrans(manager.scores_single[0].sel(time=timesel)), x="time", color="black", linestyle="--", linewidth=3, ax=ax)
                ymin = np.min(manager.scores_single[0])
                ymax = 1.5*np.max(manager.scores_single[0]) - 0.5*ymin
                ax.set_ylim([ymin,ymax])
                ax.set_xlabel("Time")
                ax.set_title(f"")
                ax.set_ylabel(score_label)
                ax.text(0,0.95,"(iv)",transform=ax.transAxes,verticalalignment="top",horizontalalignment="left")
                
                # Row 4: running maximum score functions
                ax = axes[4]
                for i_mem in asl["member_list"]:
                    ax.plot(manager.scores_single[i_mem]["time"]*tu, np.maximum.accumulate(manager.scores_single[i_mem].values), color="red", linestyle="-", linewidth=1)
                ax.plot(hist_anc["time"]*tu, np.maximum.accumulate(manager.scores_single[0].sel(time=timesel).values), color="black", linestyle="--", linewidth=2)
                ax.set_ylabel(f"Running max score")
                ax.axvline(manager.time_origins[asl["member_list"][0]]*tu, color="red")
                ax.set_ylim([ymin,ymax])
                ax.set_xlabel("Time")
                ax.text(0,0.95,"(v)",transform=ax.transAxes,verticalalignment="top",horizontalalignment="left")

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

            # Score at saturation time, as a function of perturbation
            if manager.ens.model_params["noise"]["type"] == "jump":
                fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(10,15),sharey="row",sharex="row")
                for i_frac,frac in enumerate(rmsdft["rmsdfrac"].to_numpy()[-2:]):
                    tidx = int(round(rmsdft_ensmean.sel(rmsdfrac=frac).item()))
                    print(f"{frac = }, {tidx = }")
                    scores_sat = np.array([manager.scores_single[mem].isel(time=tidx) for mem in asl["member_list"]])
                    max_scores_sat = np.array([manager.scores_single[mem].isel(time=slice(None,tidx+1)).max().item() for mem in asl["member_list"]])
                    scores_sat_scaled = (scores_sat - np.min(scores_sat))/np.ptp(scores_sat)
                    max_scores_sat_scaled = (max_scores_sat - np.min(max_scores_sat))/np.ptp(max_scores_sat)
                    for i_mem in range(len(asl["member_list"])):
                        pert = manager.ens.mem_list[asl["member_list"][i_mem]].pert_seq.isel(time=0).to_numpy()
                        ax = axes[0,i_frac]
                        ax.scatter(pert[0],pert[1],color="red",marker='o',s=200*max_scores_sat_scaled[i_mem])
                        ax.scatter(pert[0],pert[1],color="black",marker="+",linewidth=2,s=200*scores_sat_scaled[i_mem])
                        ax = axes[1,i_frac]
                        ax.scatter(np.arctan2(pert[1],pert[0]), scores_sat[i_mem], color="black", marker="+", linewidth=2)
                        ax = axes[2,i_frac]
                        ax.scatter(np.arctan2(pert[1],pert[0]), max_scores_sat[i_mem], color="red", marker="o")
                    axes[1,i_frac].axhline(manager.scores_single[0].sel(time=asl["t_split"]+tidx), color="black", linestyle="--")
                    axes[2,i_frac].axhline(manager.scores_single[0].sel(time=slice(asl["t_split"],asl["t_split"]+tidx+1)).max().item(), color="red", linestyle="--")
                    axes[0,i_frac].set_title(r"$t_{%.3f}$"%(frac))
                axes[0,0].set_ylabel("sin component")
                axes[1,0].set_ylabel(r"score($t$)")
                axes[2,0].set_ylabel(r"$max_{s\leq t}$ score($s$)")
                for col in range(axes.shape[1]):
                    axes[-1,col].set_xlabel("Phase")
                    axes[0,col].set_xlabel("cos component")
                fig.suptitle(paramdisp)
                fig.subplots_adjust(top=0.85)
                fig.savefig(join(savedir,f"pert2score_ens{i_ens}"),**svkwargs)
                plt.close(fig)

        
        return
