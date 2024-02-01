from abc import ABC,abstractmethod
import numpy as np
import matplotlib
from numbers import Number
from functools import reduce
from matplotlib import pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
svkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import xarray as xr
from sklearn.gaussian_process import GaussianProcessRegressor,kernels
import dask
import netCDF4 as nc
from numpy.random import default_rng
import pickle
import yaml
from scipy.stats import norm as spnorm
from scipy.special import logsumexp,softmax
from scipy.interpolate import RegularGridInterpolator
from scipy import optimize as spopt
import sys
import os
from os.path import join, exists
from os import mkdir, makedirs
from collections import deque
import shutil
import glob
import copy as copylib
from importlib import reload
from multiprocessing import pool as mppool

sys.path.append("../..")
import utils
from teams_manager import TEAMSManager
from ensemble_lorenz96 import Lorenz96Ensemble, Lorenz96EnsembleMember

class Lorenz96TEAMSManager(TEAMSManager):
    def score_fun_multiple(self, hist_mem, score_multiple_ancestral=None):
        score_mem = (hist_mem["x"].sel(k=0)).compute()
        #print(f"score_mem.time = {score_mem.time.to_numpy()}")
        if score_multiple_ancestral is not None:
            #print(f"score_multiple_ancestral.time = {score_multiple_ancestral.time.to_numpy()}")
            mem_t0 = score_mem["time"][0].item()
            anc_t0 = score_multiple_ancestral["time"][0].item()
            num_timesteps_to_prepend = mem_t0 - anc_t0
            if score_multiple_ancestral.time.size < num_timesteps_to_prepend:
                raise Exception("The ancestral score is not long enough")
            score_mem = xr.concat([score_multiple_ancestral.isel(time=slice(None,num_timesteps_to_prepend)), score_mem], dim="time")
        return score_mem

    def score_fun_single(self, score_multiple):
        score = score_multiple**2
        if self.algo_params["score"]["tavg"] > 1:
            score = utils.rolling_reduction(score, self.algo_params["score"]["tavg"], "mean", nanstart=True)
        #score[dict(time=slice(-2,None))] = np.nan 
        if self.algo_params["score"]["twait"] > 0:
            score[dict(time=slice(None,self.algo_params["score"]["twait"]))] = np.nan  # Time to wait between the start of an interval and a boost 
        return score

    def similarity(self, snap0, snap1):
        return (snap0 * snap1).sum()

    def create_new_pert_seq(self, parent, tidx_split, goal):
        # Generate a perturbation of the parent's pert_seq as either a random sample, or the next step of a local optimization. In the latter case, this function assumes an "anytime" nature of the optimization
        old_pert_seq = self.ens.mem_list[parent].pert_seq
        old_pert_times = old_pert_seq["time"].to_numpy()
        t_split = self.ens.mem_list[parent].time_origin + tidx_split #* self.algo_params["split_interval"]
        new_pert_times = old_pert_times.copy()
        if not (t_split in old_pert_times):
            new_pert_times = np.sort(np.union1d(old_pert_times, t_split))

        if self.ens.model_params["noise"]["type"] in ["red","white"]:
            pert_seq = xr.DataArray(
               coords={"time": new_pert_times},
               dims=["time"],
               data=0,)
            pert_seq.loc[dict(time=old_pert_times)] = old_pert_seq.to_numpy()
            num_new_perts = pert_seq.time.sel(time=slice(t_split,None)).size
            pert_seq.loc[dict(time=slice(t_split,None))] = self.rng_perturb_branch.integers(low=self.ens.model_params["seed_min"],high=self.ens.model_params["seed_max"],size=num_new_perts)
        else:
            pert_seq = xr.DataArray(
               coords={"time": new_pert_times, "component": np.arange(self.ens.model_params["noise_dim"])},
               dims=["time","component"],
               data=0.0,)
            pert_seq.loc[dict(time=old_pert_times)] = old_pert_seq.to_numpy()
            # -------- Previously: perturb at t_split AND all later times ------
            #num_new_perts = pert_seq.time.sel(time=slice(t_split,None)).size
            #pert_seq.loc[dict(time=slice(t_split,None))] = self.rng_perturb_branch.normal(size=(num_new_perts, self.ens.model_params["noise_dim"]))
            # -------- Now:  perturb ONLY at t_split --------
            # TODO: optimize!
            ancestor = self.ens.address_book[parent][0]
            descendants = np.where([addr[0] == ancestor for addr in self.ens.address_book])[0]
            print(f"descendants = {descendants}; ancestor = {ancestor}; parent = {parent}; len(memlist) = {len(self.ens.mem_list)}; len(times2split) = {len(self.times2split)}")
            contemporaries = [desc for desc in descendants if np.abs(self.times2split[desc] - t_split) < self.algo_params["split_interval"]]
            if self.algo_params["acquisition"]["locopt"] and (len(contemporaries) > self.ens.model_params["noise_dim"]):
                X = np.array([self.ens.mem_list[desc].pert_seq.sel(time=self.times2split[desc]).to_numpy() for desc in contemporaries])
                Y = np.array([self.max_scores[desc] for desc in contemporaries])
                # Build a linear or quadratic model for the score
                X = np.concatenate((np.ones((len(contemporaries),1)), X), axis=1)
                print(f"X = \n{X}")
                print(f"Y = \n{Y}")
                coef = np.linalg.solve(X.T @ X, X.T @ Y)
                # New sample is projection of coefficient vector (to achieve goal) plus random orthogonal component
                U,S,Vh = np.linalg.svd(coef[1:].reshape(-1,1), full_matrices=True)
                xrot = np.concatenate(([1.0 / S[0] * (goal - coef[0])], self.rng_perturb_branch.normal(size=self.ens.model_params["noise_dim"]-1)))
                pert_seq.loc[dict(time=t_split)] = U @ xrot

                # Maximize the probability subject to achieving the goal
                # either choose from the optimum, or sample from an exponentially tilted distribution, perhaps with a tilt determined by the goal
                pert_seq.loc[dict(time=t_split)] = coef[1:] / np.sqrt(np.sum(coef[1:]**2))
            else:
                pert_seq.loc[dict(time=t_split)] = self.rng_perturb_branch.normal(size=self.ens.model_params["noise_dim"])

        return pert_seq,t_split

    def create_warmstart_from_file(self, init_file, init_time):
        if isinstance(init_file, str):
            print(f"init_file is a string: {init_file}")
            x = xr.open_dataset(init_file, decode_times=False).load() 
        else:
            print(f"init file is something else")
            x = init_file # Allow the manager to pass the pre-opened DataArray instead of re-opening it every single time
        x0 = x.sel(time=init_time) 
        # ------------ Defunct ---------
        #x0_pert_mag = self.algo_params["perturb_start"]["original_tree"]*(len(self.ens.mem_list)==0) + self.algo_params["perturb_start"]["nonoriginal_trees"]*(len(self.ens.mem_list)>0)
        #x0["x"] += x0_pert_mag * self.rng_perturb_plant.normal(size=x["x"].sel(time=init_time).shape)
        # ------------------------------
        warmstart_info = self.ens.default_coldstart(init_time, self.algo_params["time_horizon"])
        warmstart_info["init_cond"] = x0
        if self.ens.model_params["noise"]["type"] == "white":
            warmstart_info["pert_seq"][dict(time=0)] = self.rng_perturb_plant.integers(low=self.ens.model_params["seed_min"],high=self.ens.model_params["seed_max"])
        else:
            raise Exception("Only white noise accepted")
        return warmstart_info

    def create_warmstart_from_member(self, i_parent, pert_seq):
        parent = self.ens.mem_list[i_parent]
        warmstart_info = self.ens.default_coldstart(parent.time_origin, self.algo_params["time_horizon"])
        print(f"parent time origin = {parent.time_origin}")
        warmstart_info["init_cond"] = parent.init_cond_ancestral
        warmstart_info["pert_seq"] = pert_seq

        if warmstart_info["restart_interval"] <= 0:
            raise Exception(f"restart_interval must be positive. warmstart_info = \n{warmstart_info}")
        return warmstart_info

    @classmethod
    def compute_energy_distribution(cls, manager, savefile, overwrite=False):
        # Has energy been diverted from the statistically equivalent other longitudes to the target longitude? 
        if exists(savefile) and (not overwrite):
            scores_rolled = xr.open_dataarray(savefile)
        else:
            par = manager.ens.model_params
            n_mem = len(manager.ens.mem_list)
            scores_rolled = xr.DataArray(
                    coords={"k": np.arange(par["K"]), "mem": np.arange(n_mem)},
                    dims=["k","mem"],
                    data=np.nan)
            for i_mem in range(n_mem):
                hist = manager.ens.load_member_ancestry(i_mem).load()
                score_rolled_mem = hist["x"]**2 
                if manager.algo_params["score"]["tavg"] > 1:
                    score_rolled_mem = utils.rolling_reduction(score_rolled_mem, manager.algo_params["score"]["tavg"], "mean", nanstart=True)
                scores_rolled[dict(mem=i_mem)] = score_rolled_mem.max(dim="time")
            scores_rolled.to_netcdf(savefile)
        return scores_rolled
    @classmethod
    def plot_hovmuller_perturbations(cls, manager, ancestor, savefolder, seed):
        # Plot response given perturbation. This assumes "jump" forcing
        A = manager.ens.construct_descent_matrix()
        descendants = np.where(A[ancestor])[0]

        return
    @classmethod
    def plot_response_function(cls, manager, member, savefolder, seed):
        acqstr = f"{manager.algo_params['acquisition']['global']}-{manager.algo_params['acquisition']['local']}"
        # Plot only the perturbations, not the full state, as a function of time. May be discrete points in time 
        par = manager.ens.model_params
        A = manager.ens.construct_descent_matrix()
        descendants = np.where(A[member])[0]
        asl = manager.acq_state_local[member]
        ndesc = len(descendants)
        print(f"{member = }")
        print(f"{manager.ens.address_book[member] = }")
        print(f"{A[member] = }")
        print(f"{descendants = }")
        # Plot the difference from member as a function of perturbation 
        new_perts = np.zeros((len(descendants), par["noise_dim"]))
        gains = np.zeros(len(descendants))
        for i_desc,desc in enumerate(descendants):
            new_perts[i_desc,:] = manager.ens.mem_list[desc].pert_seq.loc[dict(time=manager.times2split[desc])]
            if i_desc > 0:
                gains[i_desc] = manager.max_scores[desc] - manager.max_scores[manager.ens.address_book[desc][-2]]

        fig,axes = plt.subplots(nrows=par["noise_dim"]+2,figsize=(6,4*(par["noise_dim"]+2))) 
        # Score
        ax = axes[0]
        hdesc, = ax.plot(np.arange(ndesc), manager.max_scores[descendants], color="black", label="Descendants")
        handles = [hdesc]
        if manager.algo_params["acquisition"]["local"] == "AK":
            nexp = len(asl["expected_next_score"])
            hexp, = ax.plot(np.arange(nexp)+1, asl["expected_next_score"], color="limegreen", label="Expected score")
            hstd, = ax.plot(np.arange(nexp)+1, asl["expected_next_score"]+asl["std_next_score"], color="limegreen", linestyle='--')
            hstd, = ax.plot(np.arange(nexp)+1, asl["expected_next_score"]-asl["std_next_score"], color="limegreen", linestyle='--')
            handles.append(hexp)
        hanc = ax.axhline(manager.max_scores[member], color="black", linestyle="--", label="Ancestor")
        ax.set_ylabel("Score")
        ax.set_title(f"{acqstr} seed {seed}, family {member}")
        ax.legend(handles=handles)
        # 2 components against each other
        ax = axes[1]
        ax.plot(new_perts[:,0], new_perts[:,1], color="black", marker=".")
        ax.axhline(0, color="black", linestyle="--")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Comp 0")
        ax.set_ylabel("Comp 1")
        
        for i_comp in range(par["noise_dim"]):
            ax = axes[i_comp+2]
            ax.plot(np.arange(ndesc), new_perts[:,i_comp], color="black", marker=".")
            ax.axhline(0, color="black", linestyle="--")
            ax.set_ylabel(f"Component {i_comp}")
        fig.savefig(join(savefolder,f"response_seed{seed}_family{member}"), **svkwargs)

        plt.close(fig)

        tanc = manager.scores_single[member]
        print(f"tanc = {tanc}")

        # For local AK: heat map of GP 
        if manager.algo_params["acquisition"]["local"] == "AK" and len(descendants) > 0:
            fig,axes = plt.subplots(ncols=3,nrows=2,figsize=(18,10),sharex=True,sharey=True)
            asg = manager.acq_state_global
            X = asg["candidate_pool"]
            mean,std = asl["gp"].predict(X, return_std=True)
            ax = axes[0,0]
            ax.scatter(X[:,0],X[:,1],c=plt.cm.magma((mean-np.min(mean))/np.ptp(mean)),marker="o")
            ax.plot(new_perts[:,0],new_perts[:,1],color="limegreen",marker="*",markersize=10)
            ax.set_title(f"GP mean ({np.min(mean):.2f},{np.max(mean):.2f})")
            ax = axes[0,1]
            ax.scatter(X[:,0],X[:,1],c=plt.cm.magma((std-np.min(std))/np.ptp(std)),marker="o")
            ax.plot(new_perts[:,0],new_perts[:,1],color="limegreen",marker="*",markersize=10)
            ax.set_title(f"GP std. dev. ({np.min(std):.2f},{np.max(std):.2f})")
            ax = axes[0,2]
            acq = manager.reward_function_gaussian_process(mean,std,new_perts,manager.max_scores[descendants])
            ax.scatter(X[:,0],X[:,1],c=plt.cm.coolwarm((acq-np.min(acq))/np.ptp(acq)),marker="o")
            ax.plot(new_perts[:,0],new_perts[:,1],color="limegreen",marker="*",markersize=10)
            ax.set_title(f"{manager.algo_params['acquisition']['reward_AK']} ({np.min(acq):.2f},{np.max(acq):.2f})")
            # Plot the actual scores attained
            ax = axes[1,0]
            ax.scatter(new_perts[:,0],new_perts[:,1],color=plt.cm.magma((manager.max_scores[descendants]-np.min(manager.max_scores[descendants]))/np.ptp(manager.max_scores[descendants])),marker="o")
            ax.set_title(f"Actual scores ({np.min(manager.max_scores[descendants]):.2f},{np.max(manager.max_scores[descendants]):.2f})")
            fig.savefig(join(savefolder,f"acqfun_seed{seed}_member{member}"))
            plt.close(fig)


        # Separation rate of scores

        fig,ax = plt.subplots()
        for i_desc,desc in enumerate(descendants):
            if i_desc > 0:
                dsq = (manager.scores_single[desc] - manager.scores_single[manager.ens.address_book[desc][-2]])**2

                logdsq = np.log(dsq.where(dsq>0))
                xr.plot.plot(logdsq, x="time",ax=ax)
        ax.set_xlim(manager.scores_single[member]["time"].data[[0,-1]])
        fig.savefig(join(savefolder,f"seprate_seed{seed}_member{member}"))
        plt.close(fig)

            
        return
    @classmethod
    def plot_hovmuller_change(cls, manager, member, savefolder, seed, abbrv, paramdisp=None):
        # member = leaf; plot the sequence of modifications that led from the ancestor to the member
        n_anc = manager.algo_params["politics"]["base_size"]
        ylim = np.array([0,1.25*np.max(manager.max_scores[:n_anc])])
        vmax = np.sqrt(np.max(manager.max_scores[:n_anc]))
        mem_addr = manager.ens.address_book[member]
        ancestor = mem_addr[0]
        A = manager.ens.construct_descent_matrix()
        descendants = np.where(A[ancestor])[0]
        descendants2plot = mem_addr 

        tu = manager.ens.model_params["time_unit"]
        K = manager.ens.model_params["K"]
        tphys = manager.scores_single[ancestor]["time"].to_numpy().copy() #* tu
        tphys -= tphys[0]

        fig,axes = plt.subplots(nrows=len(descendants2plot), ncols=3, figsize=(18,(len(descendants2plot))*4), sharex=True, sharey="col")
        hist_anc = manager.ens.mem_list[ancestor].load_history_selfmade().load()
        krollfun = lambda h: h.roll(k=K//2).assign_coords(k=(h["k"] - K*(h["k"] >= K//2)).roll(k=K//2))
        # Top row: plot ancestor only 
        hist_desc_prev = hist_anc
        i_ax = -1
        for i_desc,desc in enumerate(descendants2plot):
            # Hovmuller for descendant
            i_ax += 1
            ax = axes.flat[i_ax]
            hist_desc = manager.ens.mem_list[desc].load_history_selfmade().load() #load_member_ancestry(desc).load()
            xr.plot.pcolormesh(
                    krollfun(hist_desc["x"]), 
                    x="time", y="k", ax=ax, cmap='BrBG', vmin=-vmax, vmax=vmax, add_colorbar=False)
            ax.set_title(f"score {manager.max_scores[desc]:.2f}")
            ax.axvline(x=manager.times2split[desc], color="black")
            #ax.axvline(x=manager.time_origins[desc]+manager.max_score_tidx[desc], color="black", linestyle="--")
            ax.set_xlabel("")
            if i_desc == 0:
                axes.flat[i_ax+1].axis("off")
                axes.flat[i_ax+2].axis("off")
                i_ax += 2
            else:
                # Normalized change in Hovmuller
                i_ax += 1
                ax = axes.flat[i_ax]
                hist_desc = manager.ens.mem_list[desc].load_history_selfmade().load()
                diff = hist_desc-hist_desc_prev
                dist = Lorenz96Ensemble.norm(diff) #np.sqrt((diff**2).sum(dim="k"))
                dist = dist.where(dist>0)
                logdist = np.log(dist)
                #diff[dict(time=slice(manager.max_score_tidx[desc],None))] = np.nan
                xr.plot.pcolormesh(krollfun(diff["x"]), x="time", y="k",ax=ax, cmap='BrBG', add_colorbar=False, vmin=-vmax, vmax=vmax)
                ax.axvline(x=manager.times2split[desc], color="black")
                #ax.axvline(x=manager.time_origins[desc]+manager.max_score_tidx[desc], color="black", linestyle="--")
                ax.set_title("Difference from parent")
                ax.set_xlabel("")

                # Plot scores of parent and child
                desc_prev = descendants2plot[i_desc-1]
                i_ax += 1
                ax = axes.flat[i_ax]
                hch, = xr.plot.plot(manager.scores_single[desc], ax=ax, color="red", label="New score")
                hpar, = xr.plot.plot(manager.scores_single[descendants2plot[i_desc-1]], ax=ax, color="dodgerblue", label="Previous score")
                hsp = ax.axvline(x=manager.times2split[desc], color="black", label="Split")
                ax.plot(manager.time_origins[desc]+manager.max_score_tidx[desc], manager.max_scores[desc], marker="o", markersize=16, markerfacecolor="None", markeredgecolor="red", markeredgewidth=3)
                ax.plot(manager.time_origins[desc]+manager.max_score_tidx[desc_prev], manager.max_scores[desc_prev], marker="o", markersize=16, markerfacecolor="None", markeredgecolor="dodgerblue", markeredgewidth=3)
                hth = ax.axhline(y=manager.goals[desc], color="gray", linestyle="--", label="Current level")
                if i_desc == 1:
                    ax.legend(handles=[hpar,hch,hsp,hth],loc=(0.0,1.0))
                ax.set_ylim(ylim)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_title("")

                hist_desc_prev = hist_desc.copy()
        for ax in axes.flat[-3:]:
            ax.set_xlabel("Time")
        if paramdisp is not None:
            axes.flat[1].text(0,1,paramdisp,transform=axes.flat[1].transAxes,horizontalalignment="left",verticalalignment="top")
        fig.savefig(join(savefolder,f"hovmuller_{abbrv}_{len(descendants2plot)}desc"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return

    @classmethod
    def configure_teams(cls, config_algo, model_params, rng):
        tu = model_params["time_unit"]
        algo_params = dict({
            "time_horizon": int((config_algo["time_horizon_minustadv_phys"] + config_algo["extend_horizon_flag"]*config_algo["advance_split_time_range_phys"][0])/tu),
            "split_interval": max(1,int(round(config_algo["split_interval_phys"]/tu))),
            "advance_split_time_range": np.round(np.array(config_algo["advance_split_time_range_phys"])/tu).astype(int),
            "adaptive_advance": config_algo["adaptive_advance"],
            "chunks_per_mem": config_algo["chunks_per_mem"],
            "perturb_start": config_algo["perturb_start"],
            "politics": config_algo["politics"],
            "acquisition": config_algo["acquisition"],
            "score": dict({
                "tavg": max(1,int(config_algo["score"]["tavg_phys"]/tu)),
                #"twait": max(0, int(config_algo["score"]["twait_phys"]/tu)),
                #"min_improvement_rate": config_algo["politics"]["min_improvement_rate"],
                #"min_improvement_checkpoint": config_algo["politics"]["min_improvement_checkpoint"],
                #"min_level_progression_fraction": config_algo["score"]["min_level_progression_fraction"]
                }),
            "seeddict": dict({
                "resample": rng.integers(low=model_params["seed_min"], high=model_params["seed_max"]),
                "perturb_plant": rng.integers(low=model_params["seed_min"], high=model_params["seed_max"]),
                "perturb_branch": rng.integers(low=model_params["seed_min"], high=model_params["seed_max"]),
                })
            })
        if config_algo["score"]["twait_phys"] == "default":
            algo_params["score"]["twait"] = algo_params["advance_split_time_range"][0]
        return algo_params

    @classmethod 
    def label_from_config(cls, config_algo, config_model):
        tu = config_model["time_unit"]

        # level-raising schedule 
        sched = config_algo["politics"]["level_raising_schedule"]
        if sched["type"] == "const_frac2drop":
            sched_abbrv = f"sched{sched['speed']}"
            sched_disp = r"$\kappa=%.2fN$"%(sched['speed'])
        elif sched["type"] == "const_num2drop":
            sched_abbrv = f"drop{sched['speed']}"
            sched_disp = r"$\kappa=%d$"%(sched['speed'])

        # stopping criterion: diversity
        diversity_abbrv = f"ancdiv{config_algo['politics']['min_ancestor_diversity']}"
        diversity_disp = f"Halt with $<${config_algo['politics']['min_ancestor_diversity']} ancestors"


        acqloc_str = f"evw{config_algo['acquisition']['perturb_everywhen_TEAMS']}"
        label = (
                f"TEAMS"
                f"_{acqloc_str}"
                f"_base{config_algo['politics']['base_size']}"
                f"_{diversity_abbrv}"
                f"_{sched_abbrv}"
                f"_horz{config_algo['time_horizon_minustadv_phys']}"
                f"_tavg{config_algo['score']['tavg_phys']}"
                f"_twait{config_algo['score']['twait_phys']}"
                f"_splint{config_algo['split_interval_phys']}"
                f"_adv{config_algo['advance_split_time_range_phys'][0]}to{config_algo['advance_split_time_range_phys'][1]}"
                ).replace(".","p")
        evw_str = "Persistent" if config_algo["acquisition"]["perturb_everywhen_TEAMS"] else "Transient"
        display = "\n".join([
            #r"$N=%d$, %s"%(config_algo['politics']['base_size'], sched_disp), 
            r"$\delta=%g$"%(config_algo['advance_split_time_range_phys'][0]),
            #r"pert. interval %.2f"%(config_algo['split_interval_phys']),
            #r"%s splitting"%(evw_str),
            ])
        return label,display






    
def rolling_average(da, window_size, nanstart=True):
    dt = da["time"][1].item() - da["time"][0].item()
    nshift = min(da.time.size, int(round(window_size/dt)))
    #print(f"da timeseries length = {da.time.size}, nshift = {nshift}")
    min_periods = nshift if nanstart else 1
    da_rollavg = da.rolling({"time": nshift}, min_periods=min_periods).sum() * dt 
    return da_rollavg

def teams_analysis(dns_dir,ams_dir,mankeys,config_model,config_algo,tododict):
    # Load the managers
    datadirs = dict()
    for key in mankeys:
        datadirs[key] = join(
            ams_dir, 
            f"seed{key}"
        )# .replace(".","p") # some files come from /net/hstor001.ib/...
    mandict = dict()
    A = dict()
    A1 = dict()
    simtime_ams = 0.0
    successful_mankeys = []
    for key in mankeys:
        print(f"Starting to load manager [{key}]")
        try:
            mandict[key] = pickle.load(open(join(datadirs[key],"metadata","manager"), "rb")) 
            successful_mankeys.append(key)
            simtime_ams += len(mandict[key].ens.mem_list) * mandict[key].algo_params["time_horizon"]
            A[key] = mandict[key].ens.construct_descent_matrix("all")
            A1[key] = mandict[key].ens.construct_descent_matrix(level=1)
        except EOFError:
            print(f"Manager {key} failed to save properly")

    datadirs = dict({key: datadirs[key] for key in successful_mankeys})
    mankeys = successful_mankeys

    model_params = mandict[mankeys[0]].ens.model_params
    algo_params = mandict[mankeys[0]].algo_params
    Thorz = algo_params["time_horizon"]
    Twait = algo_params["score"]["twait"]
    Tavg = algo_params["score"]["tavg"]
    Treduced = Thorz - Twait - (Tavg - 1)
    tu = mandict[mankeys[0]].ens.model_params["time_unit"]

    _,paramdisp_model = Lorenz96Ensemble.label_from_config(config_model)
    _,paramdisp_algo = Lorenz96TEAMSManager.label_from_config(config_algo, config_model)
    paramdisp = paramdisp_model + "\n" + paramdisp_algo

    # What are the first few scores?
    if tododict["plot_initial_scores"]:
        fig,ax = plt.subplots(figsize=(20,5))
        handles = []
        for i_seed,seed in enumerate(mankeys):
            color = plt.cm.Set1(i_seed)
            for i_mem in range(6):
                h, = xr.plot.plot(mandict[seed].scores_single[i_mem], x="time", color=color, label=f"Seed {seed}")
                ax.axvline(mandict[seed].time_origins[i_mem], color='black')
            handles.append(h)
        ax.legend(handles=handles)
        fig.savefig(join(ams_dir,"initial_scores"),bbox_inches="tight",pad_inches=0.2)
        plt.close(fig)

        # Is there redundancy between the initial ensembles?
        fig,axes = plt.subplots(nrows=len(mankeys),figsize=(20,6*len(mankeys)))
        for i_seed,seed in enumerate(mankeys):
            axes[i_seed].plot(mandict[seed].max_scores[:algo_params["politics"]["base_size"]])
        fig.savefig(join(ams_dir,"initial_max_scores"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

        # Plot the rank orders for the different ensembles
        fig,ax = plt.subplots()
        handles = []
        for i_seed,seed in enumerate(mankeys):
            h, = ax.plot(np.arange(algo_params["politics"]["base_size"]), np.sort(mandict[seed].max_scores[:algo_params["politics"]["base_size"]]), label=f"Seed {seed}", marker='.')
            handles.append(h)
        ax.legend(handles=handles)
        fig.savefig(join(ams_dir,"initial_max_scores_ranked"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    

    
    # Load the control simulation
    ens_dns = pickle.load(open(join(dns_dir, "output", "ens"),"rb"))

    if tododict["summarize_gains"]:
        maxgaindict = dict({"ancwise": np.zeros(len(mankeys)), "global": np.zeros(len(mankeys))})
        for i_seed,seed in enumerate(mankeys):
            manager = mandict[seed]
            adbo = manager.ens.address_book
            anc_scores = np.array([manager.max_scores[adbo[mem][0]] for mem in range(len(adbo))])
            maxgaindict["ancwise"][i_seed] = np.max(manager.max_scores - anc_scores)
            maxgaindict["global"][i_seed] = np.max(manager.max_scores) - np.max(anc_scores)
        pickle.dump(maxgaindict,open(join(ams_dir,"maxgaindict"),"wb"))



    if tododict["summarize_tail"] or tododict["plot_return_stats"]:
        # Compute the long-term statistics of the control simulation
        roll_dim = "k" # the "longitudinal" dimension along which we can roll 
        flip_dim = ""
        roll_step = 1 
        num_rolls = int(round(ens_dns.model_params["K"]/roll_step))
        symmetry_factor = num_rolls
        dns_total_time = ens_dns.mem_list[-1].term_time_list[-1] - Thorz
        print(f"{dns_total_time = }")
        mem2load_dns = len(ens_dns.mem_list) - 1
        if tododict["summarize_tail"]:
            fda = ens_dns.load_member_ancestry(mem2load_dns).sel(time=slice(Thorz,None))
            print(f"{np.min(np.diff(fda.time)) = }")
            print(f"{fda.time[[0,-1]] = }")
            mult_fun = lambda fff: mandict[mankeys[0]].score_fun_multiple(fff)
            sing_fun = lambda mf: mandict[mankeys[0]].score_fun_single(mf) 
            # augment with symmetries
            
            fconcat = [sing_fun(mult_fun(fda)).assign_coords(time=fda.time-fda.time[0].item())]
            print(f"{np.min(np.diff(fconcat[-1].time.to_numpy())) = }")
            dtfda = fda.time[:2].diff("time").item()
            duration = fda.time[-1].item() - fda.time[0].item() + dtfda
            tmax = duration
            if flip_dim != "":
                fdar = xr.zeros_like(fda)
                fdar[:] = np.flip(fda.to_numpy(), axis=fda.dims.index(flip_dim))
                fconcat.append(sing_fun(mult_fun(fdar))[0])
                tmax += duration
            for i_roll in range(1,num_rolls):
                print(f"{i_roll = } out of {num_rolls}")
                arg = sing_fun(mult_fun(fda.roll({roll_dim: i_roll*roll_step})))
                #print(f"arg before reassignment: time bounds = {arg.time[[0,-1]]}, min diff time = {np.min(np.diff(arg.time))}")
                arg = arg.assign_coords(time=tmax+arg.time-arg.time[0].item())
                #print(f"arg after reassignment: time bounds = {arg.time[[0,-1]]}, min diff time = {np.min(np.diff(arg.time))}")
                fconcat.append(arg)
                tmax += duration
                if flip_dim != "":
                    fconcat.append(
                        sing_fun(
                            mult_fun(
                                fdar.roll({roll_dim: i_roll*roll_step}))
                            .assign_coords({"time": tmax+fda.time-fda.time[0]})))
                    tmax += duration
                #print(f"{np.min(np.diff(fconcat[-1].time.to_numpy())) = }")
                #print(f"{fconcat[-1].time.to_numpy()[[0,-1]] = }")
            #print(f"Before concatting, time bounds are {[fc.time.to_numpy()[[0,-1]] for fc in fconcat]}")
            fconcat = xr.concat(fconcat, dim="time")
            #print(f"fconcat: min = {fconcat.min().item()}, max = {fconcat.max().item()}")
        else:
            fconcat = None
    
        # Plot return period curves
        TEAMSManager.tabulate_performance_metrics(mandict, ams_dir, dns_total_time*symmetry_factor, fconcat, tododict, paramdisp=paramdisp, ylim=None, bootstrap_version="basic", plot_init_sep=True, plot_init_sup=True, plot_median=False)

        
    if tododict["plot_response_flag"]:
        for seed in mankeys:
            manager = mandict[seed]

            n_mem = len(manager.ens.mem_list)
            n_anc = manager.algo_params["politics"]["base_size"]
            # a few example criteria by which to choose ancestors for display
            max_desc_scores = np.max(A[seed][:n_anc]*manager.max_scores, axis=1)
            max_improvements = max_desc_scores - manager.max_scores[:n_anc]
            # choose the best ancestor to plot, based on one of the criteria computed above (or make your own)
            fams2plot = []
            fams2plot.append(np.argsort(max_desc_scores)[-1])
            fams2plot.append(np.argsort(max_improvements)[-1])
            fams2plot = np.unique(fams2plot)
            for ancestor in fams2plot:
                Lorenz96TEAMSManager.plot_response_function(manager, ancestor, ams_dir, seed)
    
    # Plot family tree information for some selected ancestors from each manager 
    if tododict["plot_anecdotes"]:
        # set a global set of y limits

        n_anc = mandict[mankeys[0]].algo_params["politics"]["base_size"]
        max_anc_score = max([np.max(mandict[seed].max_scores[:n_anc]) for seed in mankeys])
        ylim = [0,1.5*max_anc_score]
        for seed in mankeys:
            manager = mandict[seed]
            # Plot level progression

            n_mem = len(manager.ens.mem_list)
            n_anc = manager.algo_params["politics"]["base_size"]
            # a few example criteria by which to choose ancestors for display
            anc_scores = np.max(A[seed][:n_anc])
            max_desc_scores = np.max(A[seed][:n_anc]*manager.max_scores, axis=1)
            max_improvements = max_desc_scores - manager.max_scores[:n_anc]
            # choose the best ancestor to plot, based on one of the criteria computed above (or make your own)
            fams2plot = []
            famabbrvs = []
            #fams2plot.append(np.argsort(anc_scores)[-1])
            #famabbrvs.append(f"bestanc_seed{seed}")
            #fams2plot.append(np.argsort(max_desc_scores)[-1])
            #famabbrvs.append(f'bestscore_seed{seed}')
            fams2plot.append(np.argsort(max_improvements)[-1])
            famabbrvs.append(f'bestimp_seed{seed}')
            #fams2plot = np.unique(fams2plot)
            for i_ancestor,ancestor in enumerate(fams2plot):
                if np.sum(A[seed][ancestor,:]) > 1:
                    best_desc = np.argmax(A[seed][ancestor]*manager.max_scores)
                    if tododict["spaghetti"]:
                        TEAMSManager.plot_descendant_spaghetti(manager, best_desc, ams_dir, seed, famabbrvs[i_ancestor], paramdisp, ylim=None)
                    if tododict["hovmuller"]:
                        Lorenz96TEAMSManager.plot_hovmuller_change(manager, best_desc, ams_dir, seed, famabbrvs[i_ancestor], paramdisp=paramdisp)

    if tododict["analyze_energy_flag"]:
        edist = dict()
        fig,axes = plt.subplots(nrows=len(mankeys),figsize=(5,5*len(mankeys)), sharex=True, sharey=True)
        emin,emax = np.inf,-np.inf
        for i_seed,seed in enumerate(mankeys):
            manager = mandict[seed]
            savefile = join(manager.dirs["metadata"], "scores_rolled.nc")
            edist[seed] = Lorenz96TEAMSManager.compute_energy_distribution(manager, savefile)
            emin = min(emin,edist[seed].min().item())
            emax = max(emax,edist[seed].max().item())
        bin_edges = np.linspace(emin-1e-10,emax+1e-10,20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        for i_seed,seed in enumerate(mankeys):
            ax = axes if len(mankeys)==1 else axes[i_seed]
            hist = xr.DataArray(coords={"Energy": bin_centers, "k": np.arange(model_params["K"])}, dims=["Energy","k"], data=np.nan)
            for k in range(model_params["K"]):
                hist.loc[dict(k=k)],_ = np.histogram(edist[seed].sel(k=k,mem=slice(algo_params["politics"]["base_size"],None)).to_numpy().flatten(),bins=bin_edges,density=True)
            xr.plot.pcolormesh(hist, x="k", y="Energy", ax=ax, cmap=plt.cm.magma, norm=matplotlib.colors.LogNorm(vmin=hist.where(hist>0).min().item(),vmax=hist.max().item()))
            xr.plot.plot(edist[seed].mean(dim="mem"), x="k", color="limegreen", marker="o", ax=ax)
            ax.set_xlabel("k")
            ax.xaxis.set_tick_params(which="both",labelbottom=True)
            ax.set_ylabel("Energy")
        fig.savefig(join(ams_dir, "edist"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


    return




def teams(home_dir, seed_dir, seed, config_model, config_algo, mem_per_job):
    ensemble_size_limit = 1

    # TODO make the initialization safer 

    # --------- Define and create directories -----------



    dirs_man = dict({"metadata": join(seed_dir, "metadata")})
    dirs_ens = dict({
        "output": join(seed_dir,"output"),
        "work": join(seed_dir,"work"),
        "home": home_dir,
        })

    # Load the manager to continue a paused run, or create a new manager otherwise
    if exists(join(dirs_man["metadata"], "manager")):
        manager = pickle.load(open(join(dirs_man["metadata"], "manager"), "rb"))
    elif mem_per_job > 0:
        model_params = Lorenz96Ensemble.complete_model_params(config_model)
        tu = model_params["time_unit"]

        rng = default_rng(seed=seed)
        algo_params = Lorenz96TEAMSManager.configure_teams(config_algo, model_params, rng)

        # ------- Spinup ------------
        dirs_spinup = dict({
            "output": join(seed_dir,"spinup","output"),
            "work": join(seed_dir,"spinup","work"),
            "home": home_dir,
            })
        ens_spinup = Lorenz96Ensemble.default_init(dirs_spinup, model_params, 1)
        spinup_interval = int(50/tu)
        start_info = ens_spinup.default_coldstart(0, spinup_interval)
        start_info["pert_seq"][dict(time=0)] += seed
        ens_spinup.initialize_new_member(Lorenz96EnsembleMember, start_info)
        ens_spinup.run_batch([0], np.array([1]))
        pickle.dump(ens_spinup, open(join(ens_spinup.dirs["output"], "ens_spinup"), "wb"))
        init_pool = deque()
        init_pool.appendleft((ens_spinup.mem_list[-1].term_file_list[-1], ens_spinup.mem_list[-1].term_time_list[-1].item()))
        # ------------- end spinup -----------------

        ens = Lorenz96Ensemble.default_init(dirs_ens, model_params, ensemble_size_limit)
        manager = Lorenz96TEAMSManager(dirs_man, algo_params, init_pool)
        manager.link_model(ens)
    else:
        raise Exception("No manager exists, so you called with no reason")

    # Iterate the algorithm 
    n_mem_max = mem_per_job #len(manager.ens.mem_list) + mem_per_job
    one_more_round = (len(manager.ens.mem_list) < n_mem_max)
    while one_more_round:
        manager.take_next_step(Lorenz96EnsembleMember)
        one_more_round = (len(manager.ens.mem_list) < n_mem_max) and not (manager.acq_state_global["next_action"] == "terminate")
    return 


def teams_pipeline():
    tododict = dict(
        run_teams_flag =        1,
        plot_anecdotes =        1,
        spaghetti =             1,
        hovmuller =             0,
        plot_initial_scores =   1,
        summarize_gains =       1,
        summarize_tail =        1,
        plot_return_stats =     1,
        plot_response_flag =    0,
        tally_rejections =      0,
        )

    computer = "engaging"

    maglist = np.array([3.0,3.0,1.0,0.5,0.25])
    alist = np.array([0.0] + [1.0]*4)
    tadvlist = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])
    loop_through_params = False
    params_from_sysargs = True

    seeds = [int(seed) for seed in sys.argv[3:]]
    
    # Load all YAML files here, only once
    config_teams_file = "config_teams.yml"
    config_algo = yaml.safe_load(open(config_teams_file, "r"))
    # Modify the physical model parameters
    config_dns_file = "./config_onetier.yml"  # Or a different file generated by a parameterization procedure
    config_model = yaml.safe_load(open(config_dns_file,"r"))

    max_mem_per_job = 1024

    if loop_through_params:
        evwlist = np.array([1])
        param_sets = [(
            maglist[i_mag],
            tadvlist[i_tadv],
            evw,
            alist[i_mag])
            for i_mag in range(len(maglist)) for evw in evwlist] 
    elif params_from_sysargs:
        i_mag = int(sys.argv[1]) 
        i_a = int(sys.argv[1])
        i_tadv = int(sys.argv[2])
        param_sets = [(
            maglist[i_mag],
            tadvlist[i_tadv],
            config_algo["acquisition"]["perturb_everywhen_TEAMS"],
            alist[i_a],
            )]
    else: # keep default parameters from config file 
        param_sets = [(
            config_model["noise"]["magnitude_at_wavenumber"][0],
            config_algo["advance_split_time_range_phys"][0],
            config_algo["acquisition"]["perturb_everywhen_TEAMS"],
            config_model["a"],
            )]
    for i_combo,(mag,tadv,evw,acoef) in enumerate(param_sets):
        config_model["a"] = acoef
        config_model["noise"]["magnitude_at_wavenumber"][0] = mag
        config_algo["advance_split_time_range_phys"] = [tadv,tadv]
        config_algo["acquisition"]["perturb_everywhen_TEAMS"] = evw

        print(f"\n\n------------Beginning sigma {mag}, tadv {tadv}-----------\n\n")


        config_str_model,config_disp_model = Lorenz96Ensemble.label_from_config(config_model)
        config_str_algo,config_disp_algo = Lorenz96TEAMSManager.label_from_config(config_algo,config_model)
        # Point to the DSS (direct stochastic simulation) file from which initial conditions are drawn
        if computer == "engaging":
            scratch_dir = f"/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96"
            home_dir = "/home/ju26596/rare_event_simulation/TEAMS_L96"
            dns_dir_validation = join("/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96/2024-01-31/0", config_str_model, "DNS")
        date_str = "2024-02-01"
        sub_date_str = "1"
        expt_dir = join(scratch_dir, date_str, sub_date_str)

        print(f"config_algo: \n{config_algo}")
        print(f"config_model: \n{config_model}")

        teams_dir = join(expt_dir, config_str_model, config_str_algo)
        print(f"{teams_dir = }")
        if tododict["run_teams_flag"]:
            for seed in seeds:
                seed_dir = join(teams_dir, f"seed{seed}")
                makedirs(seed_dir, exist_ok=True)
                teams(home_dir,seed_dir,seed,config_model,config_algo,max_mem_per_job)
        if exists(teams_dir):
            teams_analysis(dns_dir_validation,teams_dir,seeds,config_model,config_algo,tododict)
    return

def deep_get(dictionary, keys, default=None): # from stackoverflow
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def teams_meta_analysis(algo_dirs,meta_dir,p0fun,p1fun,p0label,p1label,p0abbrv,p1abbrv,prefix,tododict,p0inc=True,p1ofp0_opt=None):
    # From a list of multi-run algorithm output directories, compile all the performance metrics of interest and plot agains a prescribed set of parameters. 
    # params2compare: a list of dot-separated strings of dictionary keys referring to components in model_params and/or algo_params. 
        

    if tododict["compute_divs_limited"]:
        # Compute a much more refined set of error metrics
        l2norms = dict({
            "sup": [],
            "sep": [],
            "boot": [],
            "model_params": [],
            "algo_params": [],
            })
        maxgains = dict({
            "ancwise": [], # max ancestor-to-descendant boost
            "global": [], # max descendant - max ancestor
            })
        for algo_dir in algo_dirs:
            rlev = pickle.load(open(join(algo_dir,"rlev"),"rb"))
            rlsf = pickle.load(open(join(algo_dir,"rlsf"),"rb"))
            rlev_boot = pickle.load(open(join(algo_dir,"rlev_boot"),"rb"))["teams"]["split"]["sup"].sel(est="empirical")
            rlsf_boot = pickle.load(open(join(algo_dir,"rlsf_boot"),"rb"))["teams"]["split"]["sup"].sel(est="empirical")
            maxgaindict = pickle.load(open(join(algo_dir,"maxgaindict"),"rb"))
            seeds = rlev["teams"]["split"]["sep"]["seed"].to_numpy()
            cost = pickle.load(open(join(algo_dir,"cost"),"rb"))
            n_mem_teams_sup = cost["n_mem_teams"]
            n_mem_teams_sep = n_mem_teams_sup / rlev["teams"]["split"]["sep"]["seed"].size
            man0 = pickle.load(open(join(algo_dir,f"seed0","metadata","manager"),"rb"))
            l2norms["model_params"].append(man0.ens.model_params)
            l2norms["algo_params"].append(man0.algo_params)
            maxgains["ancwise"].append(maxgaindict["ancwise"])
            maxgains["global"].append(maxgaindict["global"])

            # Do the time conversions for proper comparison
            Thorz = man0.algo_params["time_horizon"]
            Twait = man0.algo_params["score"]["twait"]
            Tavg = man0.algo_params["score"]["tavg"]
            Tteams = Thorz - Twait - (Tavg - 1)
            Tdns = Thorz - Twait - (Tavg - 1)
            tu = man0.ens.model_params["time_unit"]
            lsf2rt_coords = lambda rrr,T: rrr.assign_coords(lsf=-T*tu/np.log(-np.expm1(rrr["lsf"]))).rename(lsf="rt")
            # Restrict these return periods to times greater than the horizon
            cliprt = lambda rrr,T: rrr.where(rrr["rt"] > 5*T*tu)
            def lsf2rt_vals(ppp,T):
                rrr = -T*tu/np.log(-np.expm1(ppp))
                return rrr
            def fofdict_recursive(rdict,func,args):
                for key,item in rdict.items():
                    if isinstance(item, dict):
                        fofdict_recursive(item,func,args)
                    else:
                        rdict[key] = func(item,*args)
                return
            # Take mean over seeds for probabilities
            rlsf_mean = dict()
            for tier in ["init","split"]:
                rlsf_mean[tier] = logsumexp(np.nan_to_num(rlsf["teams"][tier]["sep"].sel(est="empirical").to_numpy(),nan=-np.inf), axis=rlsf["teams"][tier]["sep"].dims.index("seed")) - np.log(len(seeds))
                rlsf_mean[tier] = np.nan_to_num(rlsf_mean[tier], nan=np.nan, posinf=np.nan, neginf=np.nan)
            print(f"{rlsf_mean = }")
            rlsf["dns"] = lsf2rt_vals(rlsf["dns"],Tdns)
            rlev["dns"] = cliprt(lsf2rt_coords(rlev["dns"],Tdns),Thorz)
            fofdict_recursive(rlsf["teams"],lsf2rt_vals,(Tteams,))
            fofdict_recursive(rlsf_mean,lsf2rt_vals,(Tteams,))
            fofdict_recursive(rlev["teams"],lsf2rt_coords,(Tteams,))
            fofdict_recursive(rlev["teams"],cliprt,(Tteams,))
            rlev_boot = lsf2rt_coords(rlev_boot,Tteams)
            rlsf_boot = lsf2rt_vals(rlsf_boot,Tteams)


            print(f"\n\nNoise {man0.ens.model_params['noise']['magnitude_at_wavenumber'][0]}, adv {man0.algo_params['advance_split_time_range'][0]*tu}")
            rlev_teams_sup = rlev["teams"]["split"]["sup"].sel(est="empirical",confint=0,side="lo")
            rlev_teams_sup = rlev_teams_sup.where(np.isfinite(rlev_teams_sup), other=rlev_teams_sup.max())
            rlev["teams"]["split"]["sep"] = xr.where(np.isfinite(rlev["teams"]["split"]["sep"]), rlev["teams"]["split"]["sep"], rlev["teams"]["split"]["sep"].max(dim="rt"))
            rlev_boot = xr.where(np.isfinite(rlev_boot), rlev_boot, rlev_boot.max(dim="rt"))
            rlev_dns = rlev["dns"].isel(bss=0).sel(est="empirical",confint=0,side="lo")
            rlsf_teams_sup = rlsf["teams"]["split"]["sup"].sel(est="empirical",confint=0,side="lo")
            rlsf_teams_sup = rlsf_teams_sup.where(np.isfinite(rlsf_teams_sup), other=np.nan)
            rlsf_teams_mean = np.nan*xr.ones_like(rlsf_teams_sup)
            print(f"{rlsf_teams_mean.dims = }")
            print(f"{rlsf['teams']['split']['sep'].dims = }")
            rlsf_teams_mean[:] = logsumexp(rlsf["teams"]["split"]["sep"].sel(est="empirical").to_numpy(), axis=rlsf["teams"]["split"]["sep"].dims.index("seed")) - np.log(len(seeds))
            rlsf_dns = rlsf["dns"].isel(bss=0).sel(est="empirical",confint=0,side="lo")

            # Compute the L2 norms
            simtime_teams_sep = Thorz * n_mem_teams_sep
            lower_time_limit = 0 * simtime_teams_sep * tu
            l2norm_sep = np.sqrt(((rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns)**2).sel(rt=slice(lower_time_limit,None)).mean(dim="rt"))
            l2norm_sup = np.sqrt(((rlev_teams_sup - rlev_dns)**2).sel(rt=slice(lower_time_limit,None)).mean().item())
            l2norm_boot = np.sqrt((rlev_boot - rlev_dns)**2).sel(rt=slice(lower_time_limit,None)).mean(dim="rt")
            l2norms["sep"].append(l2norm_sep)
            l2norms["sup"].append(l2norm_sup)
            l2norms["boot"].append(l2norm_boot)
        l2norms["sep"] = xr.concat(l2norms["sep"],dim="paramset").assign_coords(paramset=np.arange(len(algo_dirs)))
        l2norms["boot"] = xr.concat(l2norms["boot"],dim="paramset").assign_coords(paramset=np.arange(len(algo_dirs)))
        l2norms["sup"] = np.array(l2norms["sup"])
        maxgains["ancwise"] = np.array(maxgains["ancwise"])
        maxgains["global"] = np.array(maxgains["global"])
        #l2norms["sup"] = xr.concat(l2norms["sup"],dim="paramset").assign_coords(paramset=np.arange(len(algo_dirs)))

        pickle.dump(l2norms,open(join(meta_dir,"l2norms"),"wb"))
        pickle.dump(maxgains,open(join(meta_dir,"maxgains"),"wb"))

    if tododict["plot_divs_limited"]:
        l2norms = pickle.load(open(join(meta_dir,"l2norms"),"rb"))
        maxgains = pickle.load(open(join(meta_dir,"maxgains"),"rb"))
        print(f"{maxgains = }")
        p0vals2plot_nonunique = [p0fun(modpar,algpar) for (modpar,algpar) in zip(l2norms["model_params"], l2norms["algo_params"])]
        # unique-ize
        p0vals2plot = []
        for i_p0val in range(len(p0vals2plot_nonunique)):
            if not (p0vals2plot_nonunique[i_p0val] in p0vals2plot):
                p0vals2plot.append(p0vals2plot_nonunique[i_p0val])


        # TODO sort p0vals first by a, then by mag
        par_pairs = [[p0fun(modpar,algpar), p1fun(modpar,algpar)] for (modpar,algpar) in zip(l2norms["model_params"], l2norms["algo_params"])]

        fig,axes = plt.subplots(nrows=len(p0vals2plot),ncols=2,figsize=(10,3*len(p0vals2plot)),sharex=True,gridspec_kw={"hspace": 0.05, "wspace": 0.3})
        for i_p0val,p0val in enumerate(p0vals2plot):
            idx = np.array([i for i in range(len(par_pairs)) if par_pairs[i][0] == p0val])
            idx = idx[np.argsort([par_pairs[i][1] for i in idx])]
            p1vals = np.array([par_pairs[i][1] for i in idx])

            ax = axes[i_p0val,0]
            #for seed in l2norms["sep"].seed.values:
            #    ax.plot(p1vals, l2norms["sep"].isel(paramset=idx).sel(seed=seed).values, color="red", alpha=0.5) 
            handles = []
            hsep_mean, = ax.plot(p1vals, l2norms["sep"].isel(paramset=idx).mean(dim="seed").to_numpy(), color="red", linewidth=3,linestyle="-", label=r"Runwise mean",marker='o')
            handles.append(hsep_mean)
            hsep_iqr = ax.fill_between(
                    p1vals, 
                    l2norms["sep"].isel(paramset=idx).quantile(0.25,dim="seed").to_numpy(),
                    l2norms["sep"].isel(paramset=idx).quantile(0.75,dim="seed").to_numpy(),
                    facecolor="red", edgecolor="none", zorder=-1, alpha=0.4, label=r"Runwise IQR")
            handles.append(hsep_iqr)
            hsep_ci95 = ax.fill_between(
                    p1vals, 
                    l2norms["sep"].isel(paramset=idx).quantile(0.025,dim="seed").to_numpy(),
                    l2norms["sep"].isel(paramset=idx).quantile(0.975,dim="seed").to_numpy(),
                    facecolor="red", edgecolor="none", zorder=-1, alpha=0.2, label=r"Runwise 95% CI")
            handles.append(hsep_ci95)
            if False: # don't include the noisy sup (pooled) estimate
                hsup, = ax.plot(p1vals, l2norms["sup"][idx], color="black", linewidth=3, label=r"Return level RMSE pooled",marker='o') 
                hboot_ci95 = ax.fill_between(
                        p1vals, 
                        l2norms["boot"].isel(paramset=idx).quantile(0.025,dim="boot").to_numpy(),
                        l2norms["boot"].isel(paramset=idx).quantile(0.975,dim="boot").to_numpy(),
                        facecolor="gray", edgecolor="none", zorder=-2, alpha=0.2, label=r"Return level RMSE bootwise 95% CI")
                handles.append(hboot_iqr)
                hboot_iqr = ax.fill_between(
                        p1vals, 
                        l2norms["boot"].isel(paramset=idx).quantile(0.25,dim="boot").to_numpy(),
                        l2norms["boot"].isel(paramset=idx).quantile(0.75,dim="boot").to_numpy(),
                        facecolor="gray", edgecolor="none",  zorder=-1, alpha=0.4, label=r"Return level RMSE bootwise IQR")
                handles.append(hboot_ci95)
            ylim = ax.get_ylim()
            ax.set_ylim([0, ylim[1]])
            if i_p0val == 0: 
                ax.legend(handles=handles,loc=(0,1.05), title="Return level RMSE")
            if p1ofp0_opt is not None:
                ax.axvline(p1ofp0_opt[p0val], color="gray")
            ax.text(-0.2,0.5,"\n".join([r"%s = $%g$"%(p0label_comp,p0val_comp) for (p0label_comp,p0val_comp) in zip(p0label,p0val)][1:]),horizontalalignment='right',verticalalignment='center',transform=ax.transAxes)
            ax.set_xlabel("")

            ax = axes[i_p0val,1]
            handles = []
            hancwise, = ax.plot(p1vals, maxgains["ancwise"].mean(axis=1)[idx], color="green", label="Mean family gain", marker='o')
            handles.append(hancwise)
            if False:
                hglobal, = ax.plot(p1vals, maxgains["global"][idx].mean(axis=1), color="darkorange", label="Mean population gain", marker='o')
                handles.append(hglobal)
            if p1ofp0_opt is not None:
                ax.axvline(p1ofp0_opt[p0val], color="gray")
            if i_p0val == 0:
                ax.legend(handles=handles,loc=(0,1.05))
        ax.xaxis.set_tick_params(which="both",labelbottom=False)
        for col in [0,1]:
            axes[-1,col].set_xlabel(p1label)
            axes[-1,col].xaxis.set_tick_params(which="both",labelbottom=True)
            axes[-1,col].set_xlabel(p1label)
        #fig.suptitle("L2 norms",x=0.5,y=0.9,verticalalignment="bottom")
        fig.savefig(join(meta_dir,(f"performance_summary_{prefix}_{p0abbrv}_{p1abbrv}").replace(".","p")),**svkwargs)
        plt.close(fig)



    if tododict["compute_divs"]:
        results = dict({
            "err_lsfl2": [], # L2 distance between log survival functions
            "err_lsfl2_sep": [],
            "err_lsfl2_extrap": [], # L2 distance between log survival functions
            "err_lsfl2_extrap_sep": [],
            "err_quantl2": [], # L2 distance between quantile functions
            "err_quantl2_sep": [],
            "err_quantl2_extrap": [],
            "err_quantl2_extrap_sep": [],
            "err_quantl2_extrap_std": [],
            "err_quantl2_extrap_sep_std": [],
            "err_quantl2_extrap_sep_multirun": [],
            "err_quantl1": [],
            "err_quantl1_sep": [],
            "err_quantl1_extrap": [],
            "err_quantl1_extrap_sep": [],
            "err_quantl1_extrap_std": [],
            "err_quantl1_extrap_sep_std": [],
            "err_quantl1_extrap_sep_multirun": [],
            "err_relprob": [],
            "meanmaxgain": [], # improvement from maximum initial score to final split score (avg. over runs)
            "maxgain": [], # improvement from maximum initial score to final split score (avg. over runs)
            "larp_lev": [], # longest accurate return period, with accuracy measured in terms of levels 
            "larp_lsf": [], # longest accurate return period, with accuracy measured in terms of return times
            "stddev": [],
            "absskew": [],
            "skew": [],
            "bias_eqcost": [], # bias of AMS at the return period equivalent to its own running time 
            "algo_params": [],
            "model_params": [],
            })
        for algo_dir in algo_dirs:

            rlev = pickle.load(open(join(algo_dir,"rlev"),"rb"))
            rlsf = pickle.load(open(join(algo_dir,"rlsf"),"rb"))
            rlev_boot = pickle.load(open(join(algo_dir,"rlev_boot"),"rb"))
            rlsf_boot = pickle.load(open(join(algo_dir,"rlsf_boot"),"rb"))
            seeds = rlev["teams"]["split"]["sep"]["seed"].to_numpy()
            print(f"{rlsf['dns'] = }")
            cost = pickle.load(open(join(algo_dir,"cost"),"rb"))
            n_mem_teams_sup = cost["n_mem_teams"]
            n_mem_teams_sep = n_mem_teams_sup / rlev["teams"]["split"]["sep"]["seed"].size
            man0 = pickle.load(open(join(algo_dir,f"seed0","metadata","manager"),"rb"))

            tu = man0.ens.model_params["time_unit"]
            Thorz = man0.algo_params["time_horizon"]
            T = Thorz - man0.algo_params["score"]["twait"] - (man0.algo_params["score"]["tavg"] - 1)
            print(f"\n\nNoise {man0.ens.model_params['noise']['magnitude_at_wavenumber'][0]}, adv {man0.algo_params['advance_split_time_range'][0]*tu}")
            results["algo_params"].append(man0.algo_params)
            results["model_params"].append(man0.ens.model_params)
            print(f"rlev lsf = {rlev['dns']['lsf']}")
            # Quantile L2 distance
            rlev_teams = rlev["teams"]["split"]["sup"].sel(est="empirical",confint=0,side="lo")
            rlev_teams = rlev_teams.where(np.isfinite(rlev_teams), other=rlev_teams.max())
            rlev_dns = rlev["dns"].isel(bss=0).sel(est="empirical",confint=0,side="lo")
            rlsf_teams = rlsf["teams"]["split"]["sup"].sel(est="empirical",confint=0,side="lo")
            rlsf_teams = rlsf_teams.where(np.isfinite(rlsf_teams), other=np.nan)
            rlsf_teams_mean = np.nan*xr.ones_like(rlsf_teams)
            print(f"{rlsf_teams_mean.dims = }")
            print(f"{rlsf['teams']['split']['sep'].dims = }")
            rlsf_teams_mean[:] = logsumexp(rlsf["teams"]["split"]["sep"].sel(est="empirical").to_numpy(), axis=rlsf["teams"]["split"]["sep"].dims.index("seed")) - np.log(len(seeds))
            rlsf_dns = rlsf["dns"].isel(bss=0).sel(est="empirical",confint=0,side="lo")
            lsf2rt_coords = lambda rrr: rrr.assign_coords(lsf=-T*tu/np.log(-np.expm1(rrr["lsf"]))).rename(lsf="rt")
            rp = -T*tu/np.log(-np.expm1(rlev_dns.coords["lsf"]))
            # ---------- Define integrated error metric -------
            l2norm = lambda R: np.sqrt((R**2).mean().item())
            l2norm_std = lambda R: np.sqrt((R**2).mean(dim=np.setdiff1d(list(R.dims),["seed"]))).std(dim="seed").item()
            l1norm = lambda R: (np.abs(R)).mean().item()
            l1norm_std = lambda R: (np.abs(R)).mean(dim=np.setdiff1d(list(R.dims),["seed"])).std(dim="seed").item()
            # -------------------------------------------------
            results["err_lsfl2"].append(l2norm(rlsf_teams_mean - rlsf_dns))
            results["err_lsfl2_sep"].append(l2norm(rlsf["teams"]["split"]["sep"].sel(est="empirical") - rlsf_dns))
            results["err_lsfl2_extrap"].append(l2norm((rlsf_teams_mean - rlsf_dns).where(rlsf_dns<-np.log(n_mem_teams_sep))))
            results["err_lsfl2_extrap_sep"].append(l2norm((rlsf["teams"]["split"]["sep"].sel(est="empirical") - rlsf_dns).where(rlsf_dns<-np.log(n_mem_teams_sep))))
            results["err_quantl2"].append(l2norm(rlev_teams - rlev_dns))
            results["err_quantl2_sep"].append(l2norm(rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns))
            results["err_quantl2_extrap"].append(l2norm((rlev_teams - rlev_dns).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            results["err_quantl2_extrap_sep"].append(l2norm((rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            results["err_quantl2_extrap_sep_std"].append(l2norm_std((rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            results["err_quantl1"].append(l1norm(rlev_teams - rlev_dns))
            results["err_quantl1_sep"].append(l1norm(rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns))
            results["err_quantl1_extrap"].append(l1norm((rlev_teams - rlev_dns).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            results["err_quantl1_extrap_sep"].append(l1norm((rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            results["err_quantl1_extrap_sep_std"].append(l1norm_std((rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev_dns).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            results["err_relprob"].append(l1norm(1 - np.exp(rlsf_teams_mean.fillna(-np.inf) - rlsf_dns)))
            results["stddev"].append(l2norm((rlev["teams"]["split"]["sep"].sel(est="empirical") - rlev["teams"]["split"]["sep"].sel(est="empirical").mean(dim="seed"))))
            results["absskew"].append(l1norm((rlev_teams - rlev["teams"]["split"]["sep"].sel(est="empirical").mean(dim="seed")).sel(lsf=slice(-np.log(n_mem_teams_sep),None))))
            med_minus_pooled = (rlev["teams"]["split"]["sep"].sel(est="empirical").median(dim="seed") - rlev_teams).sel(lsf=slice(-np.log(n_mem_teams_sep),None))
            print(f"{med_minus_pooled = }")
            results["skew"].append((rlev["teams"]["split"]["sep"].sel(est="empirical").median(dim="seed") - rlev_teams).sel(lsf=slice(-np.log(n_mem_teams_sep),None)).mean().item())
            i_lsf_eqcost = np.where(rp <= 5*cost["n_mem_teams"]*Thorz*tu)[0][-1]
            print(f"{i_lsf_eqcost = }")
            results["bias_eqcost"].append((rlev_teams.isel(lsf=i_lsf_eqcost) - rlev_dns.isel(lsf=i_lsf_eqcost)).item())
            # Longest return period with overlapping confidence intervals
            dns_within_ci_lev = np.logical_and(
                    rlev["teams"]["split"]["sep"].sel(est="empirical").quantile(0.25,dim="seed") <= rlev_dns, 
                    rlev["teams"]["split"]["sep"].sel(est="empirical").quantile(0.75,dim="seed") >= rlev_dns
                    )
            dns_within_ci_lsf = np.logical_and(
                    2*rlsf["teams"]["split"]["sep"].sel(est="empirical").quantile(0.25,dim="seed")-1*rlsf["teams"]["split"]["sep"].sel(est="empirical").quantile(0.5,dim="seed") <= rlsf["dns"].sel(est="empirical",confint=0,side="lo"), 
                    2*rlsf["teams"]["split"]["sep"].sel(est="empirical").quantile(0.75,dim="seed")-1*rlsf["teams"]["split"]["sep"].sel(est="empirical").quantile(0.5,dim="seed") >= rlsf["dns"].sel(est="empirical",confint=0,side="lo")
                    ) * (
                    np.isfinite(rlsf["teams"]["split"]["sep"].sel(est="empirical")).sum(dim="seed") >= 2
                    )
            print(f"{dns_within_ci_lev = }")
            print(f"{dns_within_ci_lsf = }")
            dns_within_ci_lev = np.logical_or(dns_within_ci_lev, rp <= 1e3)
            dns_within_ci_lsf = np.logical_or(dns_within_ci_lsf, -T*tu/np.log(-np.expm1(rlsf["dns"].sel(est="empirical",confint=0,side="lo"))) <= 1e3)
            if np.all(dns_within_ci_lev):
                print("All in agreement")
                first_disagreement_lev = len(rp)-1
            else:
                first_disagreement_lev = np.where(dns_within_ci_lev.to_numpy() == 0)[0][0]
                print(f"Disagree at {first_disagreement_lev = }")
            if np.all(dns_within_ci_lsf):
                print("All in agreement")
                first_disagreement_lsf = len(rp)-1
            else:
                first_disagreement_lsf = np.where(dns_within_ci_lsf.to_numpy() == 0)[0][0]
                print(f"Disagree at {first_disagreement_lsf = }")
            results["larp_lev"].append(rp[first_disagreement_lev].item())
            results["larp_lsf"].append(-T*tu/np.log(-np.expm1(rlsf["dns"].isel(bss=0).sel(est="empirical",confint=0,side="lo").isel(lev=first_disagreement_lsf).item())))
            print(f"{results['larp_lsf'] = }")
            # Max gains
            gains = (rlev["teams"]["split"]["sep"].sel(est="empirical").max(dim="lsf") - rlev["teams"]["init"]["sep"].sel(est="empirical").max(dim="lsf"))
            results["maxgain"].append(gains.max(dim=("seed")).item())
            results["meanmaxgain"].append(gains.mean(dim=("seed")).item())
        pickle.dump(results,open(join(meta_dir,"metaresults"),"wb"))

    if tododict["plot_divs"]:
        results = pickle.load(open(join(meta_dir,"metaresults"),"rb"))
        par_pairs = np.array([[p0fun(modpar,algpar), p1fun(modpar,algpar)] for (modpar,algpar) in zip(results["model_params"], results["algo_params"])])
        print(f"{par_pairs = }")
        label_logscale_fd_triples = [
                (r"Relative prob. err",0,"err_relprob"),
                (r"Log-survival prob $L^2$ error (pooled)",0,"err_lsfl2"),
                (r"Log-survival prob $L^2$ error (separate)",0,"err_lsfl2_sep"),
                (r"Log-survival prob $L^2$ extrapolation error (pooled)",0,"err_lsfl2_extrap"),
                (r"Log-survival prob $L^2$ extrapolation error (separate)",0,"err_lsfl2_extrap_sep"),
                (r"Return level $L^2$ error (pooled)",0,"err_quantl2"),
                (r"Return level $L^2$ error (separate)",0,"err_quantl2_sep"),
                (r"Return level $L^2$ extrapolation error (pooled)",0,"err_quantl2_extrap"),
                (r"Return level $L^2$ extrapolation error (separate)",0,"err_quantl2_extrap_sep"),
                (r"Return level $L^2$ extrapolation error stddev",0,"err_quantl2_extrap_sep_std"),
                (r"Return level $L^1$ error (pooled)",0,"err_quantl1"),
                (r"Return level $L^1$ error (separate)",0,"err_quantl1_sep"),
                (r"Return level $L^1$ extrapolation error (pooled)",0,"err_quantl1_extrap"),
                (r"Return level $L^1$ extrapolation error (separate)",0,"err_quantl1_extrap_sep"),
                (r"Return level $L^1$ extrapolation error stddev",0,"err_quantl1_extrap_sep_std"),
                (r"Mean max score boost",0,"meanmaxgain"),
                ]
        for i_fdt,fdt in enumerate(label_logscale_fd_triples):
            fd_label,log_scale,fd = fdt
            resfd = np.array(results[fd])
            p0vals2plot = np.unique(par_pairs[:,0])
            if not p0inc:
                p0vals2plot = p0vals2plot[::-1]
            fig,axes = plt.subplots(nrows=len(p0vals2plot),figsize=(8,2*len(p0vals2plot)),sharex=True,gridspec_kw={"hspace": 0.0})
            # Multiple lines
            for i_p0val,p0val in enumerate(p0vals2plot):
                ax = axes[i_p0val]
                idx = np.where(par_pairs[:,0] == p0val)[0]
                idx = idx[np.argsort(par_pairs[idx,1])]
                ax.plot(par_pairs[idx,1],resfd[idx],marker=".",color="black")
                #yrange = np.array([np.nanmin(resfd[idx]),np.nanmax(resfd[idx])])
                #ax.set_ylim([1.1*yrange[0]-0.1*yrange[1],1.2*yrange[1]-0.2*yrange[0]])
                ax.text(-0.1,0.5,r"%s = %.2f"%(p0label,p0val),transform=ax.transAxes,horizontalalignment="right",verticalalignment="center")
                ax.set_xlabel("")
                if log_scale:
                    ax.set_yscale("log")
                #else:
                #    ax.axhline(0,color="black",linestyle="--")
                ax.xaxis.set_tick_params(which="both",labelbottom=False)
            axes[-1].set_xlabel(p1label)
            axes[-1].xaxis.set_tick_params(which="both",labelbottom=True)
            axes[-1].set_xlabel(p1label)
            fig.suptitle(fd_label,x=0.5,y=0.9,verticalalignment="bottom")
            fig.savefig(join(meta_dir,f"performance_{fd}_{prefix}_{p0abbrv}_{p1abbrv}").replace(".","p"),**svkwargs)
            plt.close(fig)
        print(f"{results['larp_lsf'] = }")

    if tododict["copy_images"]:
        plot_filenames = dict({
            "dest": dict({
                "rl_of_rt": [],
                "rt_of_rl": [],
                "hovmuller": [],
                "score_spaghetti": [],
                }),
            "source": dict({
                "rl_of_rt": [],
                "rt_of_rl": [],
                "hovmuller": [],
                "score_spaghetti": [],
                }),
            })
        for algo_dir in algo_dirs:

            man0 = pickle.load(open(join(algo_dir,f"seed0","metadata","manager"),"rb"))
            # -------- Collect saved image files -----------
            p0val = p0fun(man0.ens.model_params,man0.algo_params)
            p1val = p1fun(man0.ens.model_params,man0.algo_params)
            pstr = (f"{prefix}_{p0abbrv}{p0val}_{p1abbrv}{p1val}").replace(".","p")
            print(f"{algo_dir = }")
            seedstr = f"seeds0to55" #.join([str(i) for i in range(32)])
            plot_filenames["source"]["rl_of_rt"].append(glob.glob(join(algo_dir,"rl_of_rt_%s.png"%seedstr))[0])
            plot_filenames["dest"]["rl_of_rt"].append(join(meta_dir,f"rl_of_rt_{pstr}.png"))
            plot_filenames["source"]["rt_of_rl"].append(glob.glob(join(algo_dir,"rt_of_rl_%s.png"%seedstr))[0])
            plot_filenames["dest"]["rt_of_rl"].append(join(meta_dir,f"rt_of_rl_{pstr}.png"))
            hovmuller_files = [] #glob.glob(join(algo_dir,"hovmuller_bestimp*.png"))
            if False and len(hovmuller_files) > 0:
                plot_filenames["source"]["hovmuller"] += hovmuller_files[:1]
                plot_filenames["dest"]["hovmuller"] += [join(meta_dir,f"{os.path.basename(hf)}_{pstr}.png") for hf in hovmuller_files]
            spaghetti_files = [] #glob.glob(join(algo_dir,"spaghetti_bestimp*.png"))
            if False and len(spaghetti_files) > 0:
                plot_filenames["source"]["score_spaghetti"] += spaghetti_files[:1]
                plot_filenames["dest"]["score_spaghetti"].append(join(meta_dir,f"spaghetti_{pstr}.png"))
            # -----------------------------------------------
        for vistype in ["rt_of_rl","rl_of_rt","hovmuller","score_spaghetti"]: #plot_filenames["source"].keys():
            for i_img,img in enumerate(plot_filenames["source"][vistype]):
                shutil.copyfile(plot_filenames["source"][vistype][i_img], plot_filenames["dest"][vistype][i_img])
        
    return 

def meta_analysis_pipeline():
    # Meta-analysis
    tododict = dict({
        "plot_return_stats_multilead":   0,
        "compute_divs_limited":          0,
        "plot_divs_limited":             1,
        "compute_divs":                  0,
        "plot_divs":                     0,
        "copy_images":                   0,
        })
    computer = "engaging"
    if computer == "engaging":
        home_dir = "/home/ju26596/rare_event_simulation/TEAMS_L96"
        scratch_dir = f"/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96"
    date_str = "2024-01-31"
    sub_date_str = "1"
    expt_dir = join(scratch_dir, date_str, sub_date_str)
    print(f"{os.listdir(expt_dir) = }")

    base_size = 128
    dropstr = 'drop1'
    mag_list = np.array([3.0,3.0,1.0,0.5,0.25])
    a_list = np.array([0.0] + [1.0]*4)
    tadv_list = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0])

    meta_dir = join(expt_dir,f"meta_{dropstr}_base{base_size}")
    makedirs(meta_dir,exist_ok=True)

    delta_opt = dict({ # mapping from pairs (a,mag) to optimal advance split times
        (0.0,3.0, ): 0.0,
        (1.0,3.0, ): 0.0,
        (1.0,1.0, ): 0.6,
        (1.0,0.5, ): 1.0,
        (1.0,0.25,): 1.4,
        })

    if tododict["plot_return_stats_multilead"]:
        for (mag,a) in zip(mag_list,a_list):
            rlev_list = []
            T_list = []
            for tadv in tadv_list:
                algo_dir = glob.glob(join(expt_dir,(f"F6p0_K40_J0_a{a}_white_wave4mag{mag}/*adv{tadv}to{tadv}").replace(".","p")))[0]
                rlev_new = pickle.load(open(join(algo_dir,"rlev"),"rb"))
                print(f"\n\n{algo_dir = }\n{mag = }\n{tadv = }\n{rlev_new['teams']['split']['sup'].sel(est='empirical',confint=0,side='lo').max().item() = }\n")
                rlev_list.append(rlev_new)
                man0 = pickle.load(open(join(algo_dir,"seed0","metadata","manager"),"rb"))
                Tteams = man0.algo_params["time_horizon"] - man0.algo_params["score"]["twait"] - man0.algo_params["score"]["tavg"] + 1
                T_list.append(Tteams)
                tu = man0.ens.model_params["time_unit"]
            suffix = (f"mag{mag}_a{a}").replace(".","p")
            label = r"$F_4=$%.2f"%(mag)
            Lorenz96TEAMSManager.plot_return_stats_multiple_leads(rlev_list, T_list, tu, tadv_list, delta_opt[(a,mag)], meta_dir, suffix, label)

    forcing_dirs = [join(expt_dir,(f"F6p0_K40_J0_a{a}_white_wave4mag{sig}").replace(".","p")) for (sig,a) in zip(mag_list,a_list)]
    # keep this same order 
    algo_dirs = []
    for fd in forcing_dirs[1:]:
        print(f"{fd = }")
        print(f"{exists(fd) = }")
        algo_dirs += glob.glob(join(fd, f"AMS*evw1*base{base_size}*{dropstr}*"))
    print(f"{algo_dirs = }")

    # Define a 2D subspace of observables (of parameters) to plot
    p0fun = lambda modpar,algpar: (modpar['a'],modpar["noise"]["magnitude_at_wavenumber"][0],)
    p1fun = lambda modpar,algpar: np.round(algpar["advance_split_time_range"][0] * modpar["time_unit"] * 100) / 100
    p0label = (r"$a$",r"$F_4$") # a tuple of labels 
    p1label = r"$\delta$"
    p0abbrv = "mag"
    p1abbrv = "advxa"
    prefix = "evw1"
    teams_meta_analysis(algo_dirs,meta_dir, p0fun, p1fun, p0label, p1label, p0abbrv, p1abbrv, prefix, tododict, p0inc=False, p1ofp0_opt=delta_opt)
    return

if __name__ == "__main__": 
    teams_pipeline()
    #meta_analysis_pipeline()

