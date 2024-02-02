from abc import ABC,abstractmethod
from fractions import Fraction
import numpy as np
import matplotlib
from numbers import Number
from functools import reduce
from itertools import combinations
from matplotlib import pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
svkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import xarray as xr
import dask
import netCDF4 as nc
from numpy.random import default_rng
import pickle
import yaml
from scipy.stats import norm as spnorm
import sys
import os
from os.path import join, exists
from os import mkdir, makedirs
from collections import deque
import glob
import copy as copylib
from importlib import reload
from multiprocessing import pool as mppool

sys.path.append("../..")
import utils
from pert_manager import PERTManager
from ensemble_lorenz96 import Lorenz96Ensemble, Lorenz96EnsembleMember

class Lorenz96PERTManager(PERTManager):
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
        if self.algo_params["score"]["twait"] > 0:
            score[dict(time=slice(None,self.algo_params["score"]["twait"]))] = np.nan  # Time to wait between the start of an interval and a boost 
        return score

    def similarity(self, snap0, snap1):
        return (snap0 * snap1).sum()

    def create_warmstart_from_file(self, init_file, init_time, duration):
        if isinstance(init_file, str):
            print(f"init_file is a string: {init_file}")
            x = xr.open_dataset(init_file, decode_times=False).load() 
        else:
            print(f"init file is something else")
            x = init_file # Allow the manager to pass the pre-opened DataArray instead of re-opening it every single time
        x0 = x.sel(time=init_time) 
        warmstart_info = self.ens.default_coldstart(init_time, duration)
        warmstart_info["init_cond"] = x0
        if self.ens.model_params["noise"]["type"] == "white":
            warmstart_info["pert_seq"][dict(time=0)] = self.rng_perturb_plant.integers(low=self.ens.model_params["seed_min"],high=self.ens.model_params["seed_max"])
        else:
            raise Exception("Only white noise accepted")
        return warmstart_info

    def create_warmstart_from_member(self, i_parent, t_split, pert_seq, duration):
        parent = self.ens.mem_list[i_parent]
        warmstart_info = self.ens.default_coldstart(t_split, duration)
        print(f"{parent.time_origin = }")
        print(f"{t_split = }")
        warmstart_info["init_cond"] = self.ens.mem_list[i_parent].load_history_selfmade().sel(time=t_split).load()
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
            hexp, = ax.plot(np.arange(ndesc)+1, asl["expected_next_score"], color="limegreen", label="Expected score")
            hstd, = ax.plot(np.arange(ndesc)+1, asl["expected_next_score"]+asl["std_next_score"], color="limegreen", linestyle='--')
            hstd, = ax.plot(np.arange(ndesc)+1, asl["expected_next_score"]-asl["std_next_score"], color="limegreen", linestyle='--')
            handles.append(hexp)
        hanc = ax.axhline(manager.max_scores[member], color="black", linestyle="--", label="Ancestor")
        ax.set_ylabel("Score")
        ax.set_title(f"")
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
            fig,axes = plt.subplots(ncols=3,figsize=(18,6))
            asg = manager.acq_state_global
            X = asg["candidate_pool"]
            mean,std = asl["gp"].predict(X, return_std=True)
            ax = axes[0]
            ax.scatter(X[:,0],X[:,1],c=plt.cm.magma((mean-np.min(mean))/np.ptp(mean)),marker="o")
            ax.plot(new_perts[:,0],new_perts[:,1],color="limegreen",marker="*",markersize=10)
            ax.set_title("Mean")
            ax = axes[1]
            ax.scatter(X[:,0],X[:,1],c=plt.cm.magma((std-np.min(std))/np.ptp(std)),marker="o")
            ax.plot(new_perts[:,0],new_perts[:,1],color="limegreen",marker="*",markersize=10)
            ax.set_title("Std. Dev.")
            ax = axes[2]
            lam = (mean - np.max(manager.max_scores[descendants]) - asl["xi"])/std
            exp_imp = std * (lam*spnorm.cdf(lam) + spnorm.pdf(lam))
            ax.scatter(X[:,0],X[:,1],c=plt.cm.magma((exp_imp-np.min(exp_imp))/np.ptp(exp_imp)),marker="o")
            ax.plot(new_perts[:,0],new_perts[:,1],color="limegreen",marker="*",markersize=10)
            ax.set_title("Exp. Imp.")
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
    def plot_hovmuller_change(cls, manager, member, savefolder, seed):
        ancestor = manager.ens.address_book[member][0]
        A = manager.ens.construct_descent_matrix()
        descendants = np.where(A[ancestor])[0]
        tu = manager.ens.model_params["time_unit"]
        K = manager.ens.model_params["K"]
        tphys = manager.scores_single[ancestor]["time"].to_numpy().copy() #* tu
        tphys -= tphys[0]
        # Only plot the Hovmuller diagrams along the line of descendants leading to the best descendant
        best_desc = descendants[np.argmax(manager.max_scores[descendants])]
        best_addr = manager.ens.address_book[best_desc]

        fig,axes = plt.subplots(nrows=len(best_addr), ncols=3, figsize=(20,(len(best_addr))*6))
        hist_anc = manager.ens.load_member_ancestry(ancestor).load()
        krollfun = lambda h: h.roll(k=K//2).assign_coords(k=(h["k"] - K*(h["k"] >= K//2)).roll(k=K//2))
        #hist_anc = hist_anc.assign_coords(time=tphys)
        hist_desc_prev = hist_anc
        i_ax = 0
        for i_desc,desc in enumerate(best_addr):
            ax = axes.flat[i_ax]
            hist_desc = manager.ens.load_member_ancestry(desc).load()
            #hist_desc = hist_desc.assign_coords(time=tphys)
            xr.plot.pcolormesh(
                    krollfun(hist_desc["x"]), 
                    x="time", y="k", ax=ax)
            ax.set_title(f"Member {desc}")
            ax.axvline(x=manager.times2split[desc], color="black")
            ax.axvline(x=manager.time_origins[desc]+manager.max_score_tidx[desc], color="black")
            ax.axvline(x=manager.time_origins[desc]+manager.max_score_tidx[ancestor], color="black",linestyle="--")
            i_ax += 1
            ax = axes.flat[i_ax]
            if i_desc > 0:
                hist_desc = manager.ens.load_member_ancestry(desc).load()
                diff = hist_desc["x"]-hist_desc_prev["x"]
                dist = np.sqrt((diff**2).sum(dim="k"))
                dist = dist.where(dist>0)
                logdist = np.log(dist)
                #diff[dict(time=slice(manager.max_score_tidx[desc],None))] = np.nan
                xr.plot.pcolormesh(krollfun(diff)/dist, x="time", y="k",ax=ax)
                ax.axvline(x=manager.times2split[desc], color="black")
                ax.axvline(x=manager.time_origins[desc]+manager.max_score_tidx[desc], color="black")
                ax.axvline(x=manager.time_origins[desc]+manager.max_score_tidx[ancestor], color="black",linestyle="--")
                ax.set_title("Normalized Diff. from parent")
                # Plot distance as function of time 
                i_ax += 1
                ax = axes.flat[i_ax]
                xr.plot.plot(logdist, x="time", ax=ax, color="black")
                ax.set_xlabel("Time")
                ax.set_title("Log dist. from parent")
                ax.set_ylabel("")

                hist_desc_prev = hist_desc.copy()
            else:
                ax.axis("off")
                i_ax += 1
                ax.axis("off")
            i_ax += 1
        fig.savefig(join(savefolder,f"hovmuller_seed{seed}_family{ancestor}"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return

    @classmethod
    def configure_pert(cls, config_algo, model_params, rng):
        tu = model_params["time_unit"]
        algo_params = dict({
            "spinup_time": int(config_algo["spinup_time_phys"]/tu),
            "time_horizon_spine": int(config_algo["time_horizon_spine_phys"]/tu),
            "time_horizon_hindcast": int(config_algo["time_horizon_hindcast_phys"]/tu),
            "hindcast_init_interval": max(1,int(config_algo["hindcast_init_interval_phys"]/tu)),
            "hindcast_ensemble_size": config_algo["hindcast_ensemble_size"],
            "chunks_per_mem": config_algo["chunks_per_mem"],
            "perturb_start": config_algo["perturb_start"],
            "perturb_start": config_algo["perturb_start"],
            "score": dict({
                "tavg": max(1,int(config_algo["score"]["tavg_phys"]/tu)),
                "twait": max(0, int(config_algo["score"]["twait_phys"]/tu)),
                }),
            "seeddict": dict({
                "perturb_plant": rng.integers(low=model_params["seed_min"], high=model_params["seed_max"]),
                "perturb_branch": rng.integers(low=model_params["seed_min"], high=model_params["seed_max"]),
                })
            })
        return algo_params

    @classmethod 
    def label_from_config(cls, config_algo, config_model):
        tu = config_model["time_unit"]

        label = (
                f"PERT"
                f"_enssz{config_algo['hindcast_ensemble_size']}"
                f"_horzlong{config_algo['time_horizon_spine_phys']}"
                f"_horzshort{config_algo['time_horizon_hindcast_phys']}"
                f"_splint{config_algo['hindcast_init_interval_phys']}"
                ).replace(".","p")
        return label






    
def rolling_average(da, window_size, nanstart=True):
    dt = da["time"][1].item() - da["time"][0].item()
    nshift = min(da.time.size, int(round(window_size/dt)))
    #print(f"da timeseries length = {da.time.size}, nshift = {nshift}")
    min_periods = nshift if nanstart else 1
    da_rollavg = da.rolling({"time": nshift}, min_periods=min_periods).sum() * dt 
    return da_rollavg

def pert_analysis(pert_dir,config_model,config_algo,tododict):
    # Load the managers
    manager = pickle.load(open(join(pert_dir,"metadata","manager"), 'rb'))

    model_params = manager.ens.model_params
    algo_params = manager.algo_params

    _,paramdisp_model = Lorenz96Ensemble.label_from_config(config_model)
    paramlab_algo = Lorenz96PERTManager.label_from_config(config_algo,config_model)
    paramdisp = paramdisp_model #+ "\n" + paramdisp_algo


    if tododict["quantify_divergence_rates"]:
        Lorenz96PERTManager.quantify_divergence_rates(Lorenz96Ensemble, manager, pert_dir)
    if tododict["plot_spaghetti"]:
        Lorenz96PERTManager.plot_member_spaghetti(Lorenz96Ensemble, manager, pert_dir, paramdisp_model, state_label=r"$x_0$", score_label=r"$x_0^2$")


    return




def pert(home_dir, pert_dir, config_model, config_algo, mem_per_tree, seed):
    ensemble_size_limit = 1

    model_params = Lorenz96Ensemble.complete_model_params(config_model)
    rng = default_rng(seed=seed)
    algo_params = Lorenz96PERTManager.configure_pert(config_algo, model_params, rng)
    tu = config_model["time_unit"]

    dirs_man = dict({"metadata": join(pert_dir, "metadata")})
    dirs_ens = dict({
        "output": join(pert_dir,"output"),
        "work": join(pert_dir,"work"),
        "home": home_dir,
        })


    # Load the manager to continue a paused run, or create a new manager otherwise
    if exists(join(dirs_man["metadata"], "manager")):
        manager = pickle.load(open(join(dirs_man["metadata"], "manager"), "rb"))
    elif mem_per_tree > 0:
        # ------- Spinup ------------
        dirs_spinup = dict({
            "output": join(pert_dir,"spinup","output"),
            "work": join(pert_dir,"spinup","work"),
            "home": home_dir,
            })
        ens_spinup = Lorenz96Ensemble.default_init(dirs_spinup, model_params, 1)
        spinup_interval = int(config_algo["spinup_time_phys"]/tu)
        start_info = ens_spinup.default_coldstart(0, spinup_interval)
        start_info["pert_seq"][dict(time=0)] += seed
        ens_spinup.initialize_new_member(Lorenz96EnsembleMember, start_info)
        ens_spinup.run_batch([0], np.array([1]))
        pickle.dump(ens_spinup, open(join(ens_spinup.dirs["output"], "ens_spinup"), "wb"))
        init_pool = deque()
        init_pool.appendleft((ens_spinup.mem_list[-1].term_file_list[-1], ens_spinup.mem_list[-1].term_time_list[-1].item()))
        # ------------- end spinup -----------------
        ens = Lorenz96Ensemble.default_init(dirs_ens, model_params, ensemble_size_limit)
        manager = Lorenz96PERTManager(dirs_man, algo_params, init_pool)
        manager.link_model(ens)
    else:
        raise Exception("No manager exists, so you called with no reason")

    # Iterate the algorithm 
    one_more_round = (len(manager.ens.mem_list) < mem_per_tree) and not (manager.acq_state_global["next_action"] == "terminate")
    while one_more_round:
        manager.take_next_step(Lorenz96EnsembleMember)
        one_more_round = (len(manager.ens.mem_list) < mem_per_tree) and not (manager.acq_state_global["next_action"] == "terminate")
    return pert_dir

def pert_pipeline():
    tododict = dict(
        run_pert_flag =              1,
        quantify_divergence_rates =  1,
        plot_spaghetti =             1,
        )

    params_from_sysargs = True


    # Load all YAML files here, only once
    config_pert_file = "config_pert.yml"
    config_algo = yaml.safe_load(open(config_pert_file, "r"))
    config_dns_file = "./config_onetier.yml"  # Or a different file generated by a parameterization procedure
    config_model = yaml.safe_load(open(config_dns_file,"r"))

    computer = "engaging"
    if computer == "engaging":
        home_dir = "/home/ju26596/rare_event_simulation/TEAMS_L96"
        scratch_dir = f"/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96"

    date_str = "2024-02-01"
    sub_date_str = "2"
    expt_dir = join(scratch_dir, date_str, sub_date_str)
    max_mem_per_tree = 1000
    max_mem_per_tree = min(
            config_algo["hindcast_ensemble_size"] * int((max_mem_per_tree-1)/config_algo["hindcast_ensemble_size"]) + 1, 
            config_algo["hindcast_ensemble_size"] * (int(config_algo["time_horizon_spine_phys"] / config_algo["hindcast_init_interval_phys"]) - 1)
            )
    print(f"{max_mem_per_tree = }")


    # Modify the physical model parameters
    wn_list = np.array([1,4,7,10])
    mag_list = np.array([3.0,1.0,0.5,0.25])
    if params_from_sysargs:
        config_model['noise']['wavenumbers'][0] = wn_list[int(sys.argv[1])]
        config_model["noise"]["magnitude_at_wavenumber"][0] = mag_list[int(sys.argv[2])]
    config_label_model,config_display_model = Lorenz96Ensemble.label_from_config(config_model)
    config_label_algo = Lorenz96PERTManager.label_from_config(config_algo, config_model)
    pert_dir = join(expt_dir, config_label_model, config_label_algo)
    makedirs(pert_dir, exist_ok=True)
    seed = 0
    if tododict["run_pert_flag"]:
        pert(home_dir, pert_dir, config_model, config_algo,max_mem_per_tree, seed)

    pert_analysis(pert_dir,config_model,config_algo,tododict)
    return

def deep_get(dictionary, keys, default=None): # from stackoverflow
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def meta_analysis_pipeline():
    computer = "engaging"
    if computer == "engaging":
        home_dir = "/home/ju26596/rare_event_simulation/TEAMS_L96"
        scratch_dir = f"/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96"
    date_str = "2024-02-01"
    sub_date_str = "2"
    expt_dir = join(scratch_dir, date_str, sub_date_str)

    wavenumbers = [1,4,7,10]
    magnitudes = [3.0,1.0,0.5,0.25]
    forcing_dirs = []
    labels = []
    params2compare = dict(wavenumber=[], magnitude=[])
    for wn in wavenumbers:
        for mag in magnitudes:
            forcing_dirs.append(join(expt_dir,(f"F6p0_K40_J0_a1p0_white_wave{wn}mag{mag}").replace(".","p")))
            labels.append(r"$(m,F_m)=(%d,%.1e)$"%(wn,mag))
            params2compare["wavenumber"].append(wn)
            params2compare["magnitude"].append(mag)
    for parname in params2compare.keys():
        params2compare[parname] = np.array(params2compare[parname])
    print(f"{forcing_dirs[0] = }")
    algo_dirs = []
    for fd in forcing_dirs:
        algo_dirs += glob.glob(join(fd, "PERT*enssz16*"))
    print(f"{forcing_dirs = }")
    print(f"{algo_dirs = }")
    print(f"{labels = }")
    meta_dir = join(expt_dir,"meta_enssz16")
    makedirs(meta_dir,exist_ok=True)

    # Define parameters of interest to plot dependence on
    p0fun = lambda modpar,algpar: modpar["noise"]["wavenumbers"][0]
    p1fun = lambda modpar,algpar: modpar["noise"]["magnitude_at_wavenumber"][0]
    p0label = r"$m$"
    p1label = r"$F_m$"
    p0abbrv = "wn"
    p1abbrv = "mag"
    mag_delta_pairs = np.array([[3.0,1.0,0.5,0.25],[0.0,0.6,1.0,1.4]])
    pert_meta_analysis(algo_dirs,meta_dir,p0fun,p1fun,p0label,p1label,p0abbrv,p1abbrv,"",mag_delta_pairs)
    return

def pert_meta_analysis(algo_dirs,meta_dir,p0fun,p1fun,p0label,p1label,p0abbrv,p1abbrv,prefix,mag_delta_pairs):
   
    results = dict({
        "sat_time": [],
        "algo_params": [],
        "model_params": [],
        })
    for i_ad,ad in enumerate(algo_dirs):
        man = pickle.load(open(join(ad,"metadata","manager"),"rb"))
        results["algo_params"].append(man.algo_params)
        results["model_params"].append(man.ens.model_params)
        results["sat_time"].append(xr.open_dataarray(join(ad, "rmsdft.nc")))
    tu = results["model_params"][0]["time_unit"]
    results["sat_time"] = xr.concat(results["sat_time"], dim="param").assign_coords(param=range(len(algo_dirs)))
    par_pairs = np.array([[p0fun(modpar,algpar), p1fun(modpar,algpar)] for (modpar,algpar) in zip(results["model_params"], results["algo_params"])])

    # 0. Plot saturation time as function of parameters
    label_qoi_pairs = []
    for frac in results["sat_time"]["rmsdfrac"]:
        print(frac)
        F = Fraction(frac.item())
        print(f'{F = }')
        label_qoi_pairs.append((r"Time to %d/%d of RMSD"%(F.numerator,F.denominator),frac.item()))
    for i_lq,lq in enumerate(label_qoi_pairs):
        fig,ax = plt.subplots()
        handles = []
        for i_p0val,p0val in enumerate(np.unique(par_pairs[:,0])):
            idx = np.where(par_pairs[:,0] == p0val)[0]
            idx = idx[np.argsort(par_pairs[idx,1])]
            print(f"{idx = }")
            h = ax.errorbar(
                    par_pairs[idx,1],
                    tu*results["sat_time"].sel(rmsdfrac=lq[1]).isel(param=idx).mean(dim="ensemble").to_numpy(),
                    yerr=tu*results["sat_time"].sel(rmsdfrac=lq[1]).isel(param=idx).std(dim="ensemble").to_numpy(),
                    label=r"%s = %g"%(p0label,p0val),marker=".", capsize=3
                    )
            handles.append(h)
        h, = ax.plot(mag_delta_pairs[0,:],mag_delta_pairs[1,:],marker="o",linestyle="--",color="black",linewidth=2.5,label=r"Optimal $\delta$ for $m=4$")
        handles.append(h)
        ax.legend(handles=handles)
        ax.set_xlabel(p1label)
        ax.set_ylabel(lq[0])
        fig.savefig(join(meta_dir,(f"sattime_rmsdf{lq[1]:.2f}").replace(".","p")),**svkwargs)
        plt.close(fig)


    # 1. Plot the time to a given fraction of saturation, as a function of the fraction
    for (ia,ib,palab,pblab,paabbrv,pbabbrv) in [(0,1,p0label,p1label,p0abbrv,p1abbrv),(1,0,p1label,p0label,p1abbrv,p0abbrv)]:
        palist = np.unique(par_pairs[:,ia])
        fig,axes = plt.subplots(ncols=len(palist), nrows=1, figsize=(5*len(palist),5), sharey=True)
        for i_paval,paval in enumerate(palist):
            ax = axes[i_paval]
            idx_par = np.where(par_pairs[:,ia] == paval)[0]
            handles = []
            for i_par in idx_par:
                pbval = par_pairs[i_par,ib]
                h = ax.errorbar(results["sat_time"].isel(param=i_par)["rmsdfrac"], tu*results["sat_time"].isel(param=i_par).mean(dim="ensemble"), yerr=tu*results["sat_time"].isel(param=i_par).std(dim="ensemble"), label=r"%s$=$%s"%(pblab,str(pbval)), capsize=3)
                handles.append(h)
            ax.legend(handles=handles)
            ax.set_xscale("log")
            ax.set_title(r"%s$=$%s"%(palab,str(paval)))
            ax.set_xlabel("Fraction of saturation")
            ax.set_ylabel("Time since split" if i_paval==0 else "")
            ax.yaxis.set_tick_params(which="both",labelbottom=True)
            ax.set_xticks(results["sat_time"]["rmsdfrac"])
            ax.set_xticklabels([f"{f:.3f}" for f in results["sat_time"]["rmsdfrac"].to_numpy()])
        fig.savefig(join(meta_dir,(f"sattime_vs_{paabbrv}").replace(".","p")),**svkwargs)
        plt.close(fig)

    return





if __name__ == "__main__": 
    pert_pipeline()
    #meta_analysis_pipeline()

