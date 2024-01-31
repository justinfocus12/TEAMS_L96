
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
svkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from matplotlib.gridspec import GridSpec
from numpy.random import default_rng
from scipy.linalg import solve_sylvester
from scipy.special import softmax, logsumexp
from scipy.stats import norm as spnorm
import xarray as xr
import yaml
import os
from os.path import join, exists
from os import mkdir, makedirs
from numpy.random import default_rng
import pickle
import copy as copylib

import sys
sys.path.append("../..")
from ensemble import Ensemble,EnsembleMember
import utils


class Lorenz96Ensemble(Ensemble):
    def setup_model(self):
        return
    def load_member_ancestry(self, i_mem_leaf):
        hist_list = []
        ds = self.mem_list[i_mem_leaf].load_history_selfmade()
        t0 = ds["time"][0].item() #ds.attrs["t0"]
        for i_mem_twig in self.address_book[i_mem_leaf][::-1][1:]:
            ds_new = self.mem_list[i_mem_twig].load_history_selfmade()
            t0_new = ds_new["time"][0].item() #ds_new.attrs["t0"]
            num_timesteps_to_prepend = t0 - t0_new
            ds = xr.concat([ds_new.isel(time=slice(None,num_timesteps_to_prepend)), ds], dim="time")
            #ds.attrs["t0"] = ds_new.attrs["t0"]
            t0 = t0_new
        return ds
    @classmethod
    def norm(cls, ds):
        norm_ds = np.sqrt((ds["x"]**2).mean(dim="k"))
        return norm_ds
    @classmethod
    def label_from_config(cls, config):
        # Generate a unique directory label given a config dictionary
        fkjstr = str(f"F{config['F']}_K{config['K']}_J{config['J']}_a{config['a']}".replace(".","p"))
        fkjdisp = r"$K=%i,\ a=%f,\ F_0=%.1f$"%(config['K'], config['a'], config['F'])
        noise_str = config['noise']['type'] + "_" + "-".join([
            f"wave{config['noise']['wavenumbers'][i_wave]}mag{config['noise']['magnitude_at_wavenumber'][i_wave]}" for i_wave in range(len(config['noise']['wavenumbers']))])
        noise_display = r"" if config['a']==1 else r"$a=%g$,"%(config['a'])
        noise_display += ",".join([
            r"$F_{%d}=%g$"%(config['noise']['wavenumbers'][i_wave],config['noise']['magnitude_at_wavenumber'][i_wave]) 
            for i_wave in range(len(config['noise']['wavenumbers']))
            ])
        config_label = (
                f"{fkjstr}"
                f"_{noise_str}"
                ).replace(".","p")
        config_display = (
                f"{noise_display}"
                )
        return config_label,config_display


    @classmethod
    def complete_model_params(cls, model_params):
        # Build the Transfer matrix to convert a vector of perturbations to state space
        model_params["noise_dim"] = 2*len(model_params["noise"]["wavenumbers"]) + len(model_params["noise"]["sites"])
        model_params["state_dim"] = model_params["K"] * (1 + model_params["J"])
        model_params["sigma"] = np.zeros((model_params["state_dim"],model_params["noise_dim"]))
        component = 0
        for i_f,f in enumerate(model_params["noise"]["wavenumbers"]):
            model_params["sigma"][:model_params["K"],component] = model_params["noise"]["magnitude_at_wavenumber"][i_f] * np.cos(np.linspace(0,2*np.pi*f,model_params["K"]+1))[:-1]
            component += 1
            model_params["sigma"][:model_params["K"],component] = model_params["noise"]["magnitude_at_wavenumber"][i_f] * np.sin(np.linspace(0,2*np.pi*f,model_params["K"]+1))[:-1]
            component += 1
        for i_k,k in enumerate(model_params["noise"]["sites"]):
            model_params["sigma"][k,component] = model_params["noise"]["magnitude_at_site"][i_k]
            component += 1

        model_params["step_method"] = "euler"

        return model_params

    @classmethod
    def default_init(cls, dirs_ens, config_model, ensemble_size_limit):

        model_params = cls.complete_model_params(config_model)
       
        ensemble_size_limit = ensemble_size_limit
        ens = cls(dirs_ens, model_params, ensemble_size_limit)
        return ens
    def default_coldstart(self, init_time, restart_interval):
        coldstart_info = dict({
            "init_time": init_time, 
            "init_cond": xr.Dataset(
                data_vars={
                    "x": xr.DataArray(
                        coords = {"k": np.arange(self.model_params["K"])},
                        dims = ["k"], 
                        data = self.model_params["F"] + 0.001*np.sin(np.linspace(0,2*np.pi,self.model_params["K"])),
                        ),
                    "y": xr.DataArray(
                        coords = {"k": np.arange(self.model_params["K"]), "j": np.arange(self.model_params["J"])},
                        dims = ["k","j"], 
                        data = np.zeros((self.model_params["K"],self.model_params["J"]))
                        ),
                    }
                ),
            "time_origin": init_time,
            "restart_interval": restart_interval,
            })
        if self.model_params["noise"]["type"] in ["red","white"]:
            coldstart_info["pert_seq"] = xr.DataArray(
                coords={"time": [init_time]},
                dims=["time"],
                data=self.model_params["seed_min"], # the random seed to kick things off
                )
            coldstart_info["rng"] = default_rng(self.model_params["seed_min"])
        else:
            coldstart_info["pert_seq"] = xr.DataArray(
                    coords={"time": (init_time + np.arange(restart_interval)).astype(int), "component": np.arange(self.model_params["noise_dim"])},
                    dims=["time","component"],
                    data=0.0,
                    )
        return coldstart_info

    def load_history_selfmade_batch(self):
        n_mem = len(self.mem_list)
        files2load = []
        for i_mem in range(n_mem):
            files2load.append(self.mem_list[i_mem].hist_file_list)
        preprocess = lambda ds_imem: ds_imem.assign_coords({"time": ds_imem["time"] - ds_imem["time"][0].item()})
        ds = xr.open_mfdataset(files2load, preprocess=preprocess, combine='nested', concat_dim=['member',None], parallel=True)
        ds = ds.assign_coords({"member": np.arange(n_mem)})
        ds.close()
        return ds

class Lorenz96EnsembleMember(EnsembleMember):
    def set_run_params(self, model_params, warmstart_info, **kwargs):

        self.par = model_params.copy()
        self.par["restart_interval"] = warmstart_info["restart_interval"]

        # Warmstart information (including perturbation)
        self.init_cond_ancestral = warmstart_info["init_cond"]
        self.init_time_ancestral = warmstart_info["init_time"]
        self.init_cond = self.init_cond_ancestral
        self.init_time = self.init_time_ancestral
        self.time_origin = warmstart_info["time_origin"]

        self.term_file_list = []
        self.term_time_list = []
        self.hist_file_list = [] # This is special to low-dimensional process, where we can save out all the history we need in single files. In more complex models the information will be stored differently.

        self.pert_seq = warmstart_info["pert_seq"].copy()
        self.rng = warmstart_info["rng"]
        #print(f"self.pert_seq = \n{self.pert_seq}")

        # TODO the pert_log_density doesn't make sense when the pert_seq is random seeds
        self.pert_log_density = -0.5 * (self.pert_seq**2).sum().item()

        return

    def setup_directories(self):
        # Nothing to do here: no structure needed beyond the base directories 
        return

    def cleanup_directories(self):
        os.rmdir(self.dirs["work"])
        return

    def tendency_xx(self,x):
        # to be applied to either a single vector or a (N,K) array
        xdot = self.par["F"] + self.par["a"]*(np.roll(x,-1,axis=-1) - np.roll(x,2,axis=-1)) * np.roll(x,1,axis=-1) - x
        return xdot

    def tendency_xy_parameterized(self,x):
        # to be applied to a single vector
        xdot = self.tendency_xx(x) + np.power.outer(x,np.arange(len(self.par["polycoef"]))) @ self.par["polycoef"]
        return xdot

    def tendency_full(self,xy):
        x = xy[:self.par["K"]]
        y = xy[self.par["K"]:]
        xydot = np.zeros_like(xy)
        xydot[:self.par["K"]] = (
                self.par["F"] 
                + self.par["a"]*(np.roll(x,-1) - np.roll(x,2)) * np.roll(x,1) - x
                - self.par["h"]*self.par["c"]/self.par["b"] * np.sum(np.reshape(y,(self.par["K"],self.par["J"])), axis=1)
                )
        xydot[self.par["K"]:] = (
                self.par["c"]*self.par["b"] * (np.roll(y,1) - np.roll(y,-2)) * np.roll(y,-1) 
                - self.par["c"] * y
                + self.par["h"]*self.par["c"]/self.par["b"] * x[(np.arange(self.par["K"]*self.par["J"])/self.par["J"]).astype(int)]
                )
        return xydot

    def run_one_cycle(self, verbose=True):
        if verbose: print(f"Beginning: \n{self.rng.__getstate__() = }")
        Nt_save = self.par["restart_interval"] + 1 
        t_save = self.init_time + np.arange(Nt_save) 
        x_save = np.nan*np.ones((Nt_save,self.par["state_dim"]))
        dt = self.par["time_unit"]/self.par["sims_per_unit"]
        sqrtdt = np.sqrt(dt)

        #Nt_sim = (Nt_save-1)*self.par["sims_per_unit"]
        Nt_sim = (Nt_save-1)*self.par["sims_per_unit"] 

        # Construct the forcing sequence
        forcing = np.zeros((Nt_sim, self.par["state_dim"]))

        if self.par["noise"]["type"] in ["white","red"]:
            if verbose: print(f"About to generate random forcing")
            reseed_times = self.pert_seq["time"].to_numpy().astype(int) # TODO make it int already
            if verbose: 
                print(f"len(reseed_times) = {len(reseed_times)}")
                print(f"{reseed_times = }")
                print(f"{Nt_sim = }")
            if len(reseed_times) == 0:
                forcing[:] = sqrtdt * self.rng.normal(size=(Nt_sim, self.par["noise_dim"])) @ self.par["sigma"].T
            else:
                for i_pert,t_pert in enumerate(reseed_times): 
                    # generate white noise forcing sequence between one reseed time and the next
                    i_sim_first = (t_pert - reseed_times[0])*self.par["sims_per_unit"]
                    if i_pert == len(self.pert_seq["time"]) - 1:
                        i_sim_last = Nt_sim  # +1 ?? 
                    else:
                        i_sim_last = (reseed_times[i_pert+1] - reseed_times[0])*self.par["sims_per_unit"]
                    if verbose: print(f"{self.pert_seq.isel(time=i_pert) = }")
                    self.rng = default_rng(seed=self.pert_seq.isel(time=i_pert).item())
                    sszz = (i_sim_last-i_sim_first, self.par["noise_dim"])
                    if verbose: print(f"{sszz = }")
                    sZ = self.rng.normal(size=sszz) @ self.par["sigma"].T # s stands for scaled (or sigma)
                    forcing[i_sim_first:i_sim_last,:] = sqrtdt * sZ 

        elif self.par["noise"]["type"] == "jump":
            forcing[self.par["sims_per_unit"] * (self.pert_seq["time"]-self.init_time), :] = self.pert_seq.to_numpy() @ self.par["sigma"].T
        

        # Store datasets in which to write derivative information
        x_save[0,:self.par["K"]] = self.init_cond["x"].to_numpy()
        x_save[0,self.par["K"]:] = self.init_cond["y"].to_numpy().flatten()
        #x_save[0] += forcing[0]
        if verbose: print(f"{self.init_cond = }")
        # Decide which tendency to use
        if self.par["J"] > 0:
            tendency_fun = self.tendency_full
        elif len(self.par["polycoef"]) > 0:
            tendency_fun = self.tendency_xy_parameterized
        else:
            tendency_fun = self.tendency_xx

        #sigma_dW = sqrtdt*self.par["sigma"]*self.rng_run.normal(size=(self.par["sims_per_unit"]*Nt_save,self.par["K"]))
        i_sim_global = 0
        for i_save in range(1,Nt_save):
            x = x_save[i_save-1].copy()
            t = self.par["time_unit"] * (self.init_time + i_save - 1)
            for i_sim in range(self.par["sims_per_unit"]):
                k1x = dt*tendency_fun(x)
                if self.par["step_method"] == "euler":
                    x += k1x
                elif self.par["step_method"] == "rk2":
                    k2x = dt*tendency_fun(x + k1x/2)
                    x += (k1x + k2x) / 2
                elif self.par["step_method"] == "rk4":
                    k2x = dt*tendency_fun(x+k1x/2)
                    k3x = dt*tendency_fun(x+k2x/2)
                    k4x = dt*tendency_fun(x+k3x)
                    x += (k1x + 2*(k2x + k3x) + k4x) / 6
                x += forcing[i_sim_global]
                t += dt
                i_sim_global += 1
            x_save[i_save] = x
            if verbose and (i_save % self.par["print_interval"] == 0):
                print(f"Integrated through time {i_save*self.par['time_unit']} out of {Nt_save*self.par['time_unit']}; x(k=0,t) = {x_save[i_save,0]}")
        # Save history
        hist_file = join(self.dirs["output"], f"history_{self.init_time}-{t_save[-1]}.nc")
        ds_save = xr.Dataset(data_vars={
            "x": xr.DataArray(
                coords={"time": t_save, "k": np.arange(self.par["K"]), },
                dims=["time","k"],
                data=x_save[:,:self.par["K"]],
                ),
            "y": xr.DataArray(
                coords={"time": t_save, "k": np.arange(self.par["K"]), "j": np.arange(self.par["J"])},
                dims=["time","k","j"],
                data=np.reshape(x_save[:,self.par["K"]:], (len(t_save),self.par["K"],self.par["J"])),
                ),
            })
        ds_save.to_netcdf(hist_file)
        # Save restart
        term_file = join(self.dirs["output"], f"restart_{t_save[-1]}.nc")
        ds_restart = ds_save.isel(time=slice(-2,None))
        ds_restart.to_netcdf(term_file)
        # Update the lists
        self.term_file_list += [term_file]
        self.term_time_list += [t_save[-1]]
        self.hist_file_list += [hist_file]
        # Update state for the next round
        self.init_time += Nt_save - 1
        self.init_file = term_file

        if verbose: print(f"Ending: \n{self.rng.__getstate__() = }")

        return

    def load_history_selfmade(self):
        try:
            ds = xr.open_mfdataset(self.hist_file_list, decode_times=False)
            ds.close()
        except:
            print("WARNING the self-history file was mysteriously deleted!")
            ds = xr.Dataset(
                data_vars={
                    "x": xr.DataArray(
                        coords = {"k": np.arange(self.par["K"]), "time": np.arange(self.init_time_ancestral, self.init_time)},
                        dims = ["time","k"], 
                        data = np.nan,
                        ),
                    "y": xr.DataArray(
                        coords = {"k": np.arange(self.par["K"]), "j": np.arange(self.par["J"]), "time": np.arange(self.init_time_ancestral, self.init_time)},
                        dims = ["time","k","j"], 
                        data = np.nan,
                        ),
                    }
                )
        return ds

def compare_invariant_measures(ensdict, savedir):
    # Plot and quantitatively assess differences between invariant measures, with special attention paid to the tails 
    return

def run_spinup(home_dir, expt_dir, model_params, rng, duration_phys):
    # Deliver a physically realistic end condition
    pass


def run_control(home_dir, dns_dir, config_model, duration_phys, split_interval_phys):
    # Control simulation with fully resolved x and y variables
    # either extend an existing run or start a new one

    # -------------- Specify model parameters -----------------
    print(f"config_model = {config_model}")
    restart_interval = int(round(duration_phys / config_model["time_unit"]))
    config_label,_ = Lorenz96Ensemble.label_from_config(config_model)
    tu = config_model["time_unit"]
    split_interval = max(1, int(split_interval_phys/tu))
    print(f"dns_dir = {dns_dir}")
    
    dirs_ens = dict({
        "work": join(dns_dir,"work"),
        "output": join(dns_dir,"output"),
        "home": home_dir,
        })

    ensemble_size_limit = 1
    num_chunks = 1


    if exists(join(dirs_ens["output"],"ens")):
        ens_dns = pickle.load(open(join(dirs_ens["output"],"ens"),"rb"))
        mem_prev = ens_dns.mem_list[-1]
        start_info = ens_dns.default_coldstart(mem_prev.init_time, restart_interval)
        start_info["init_cond"] = xr.open_dataset(mem_prev.init_file).isel(time=-1)
        start_info["rng"] = mem_prev.rng
        # Remove the reseed from the pert_seq
        start_info["pert_seq"] = xr.DataArray(
            coords={"time": []},
            dims=["time"],
            data=0, # the random seed to kick things off
            )
        i_parent = len(ens_dns.mem_list)-1
    else:
        print(f"{join(dns_dir,'output','ens')} doesn't exist!")
        ens_dns = Lorenz96Ensemble.default_init(dirs_ens, config_model, 1) 
        start_info = ens_dns.default_coldstart(0, restart_interval)
        i_parent = None

        # ------------
    ens_dns.initialize_new_member(Lorenz96EnsembleMember, start_info, i_parent=i_parent)
    memidx2run = np.array([len(ens_dns.mem_list)-1])
    ens_dns.run_batch(memidx2run, np.array([num_chunks]))
    pickle.dump(ens_dns, open(join(ens_dns.dirs["output"],"ens"),"wb"))

    return

def run_stochastic_pmtzn_ensemble(dns_dir, pmtzn_dir, sens_supdir, num_new_mem, seed_init, spinup_interval_phys, duration_phys):
    # Start an ensemble branchnig off from a direct simulation but with new random forcing 
    ensemble_size_limit = 1
    ens_dns = pickle.load(open(join(dns_dir,"output","ens"),"rb"))
    tu = ens_dns.model_params["time_unit"]
    spinup_interval = int(spinup_interval_phys/tu)
    duration = int(duration_phys/tu)
    hist_dns = ens_dns.load_member_ancestry(len(ens_dns.mem_list)-1)

    model_params = pickle.load(open(join(pmtzn_dir, "model_params_pmtzn"), "rb"))
    config_label,_ = Lorenz96Ensemble.label_from_config(model_params)
    sens_dir = join(sens_supdir, f"{config_label}_ensemble_dur{duration}")
    dirs_ens = dict({
        "work": join(sens_dir,"work"),
        "output": join(sens_dir,"output"),
        "home": ens_dns.dirs["home"],
        })
    makedirs(sens_dir, exist_ok=True)
    # Create a simple 'manager' (just a dictionary)
    if exists(join(sens_dir,"manager")):
        manager = pickle.load(open(join(sens_dir,"manager"),"rb"))
    else:
        manager = dict({
            "ens": Lorenz96Ensemble.default_init(dirs_ens, model_params, ensemble_size_limit),
            "rng": default_rng(seed=seed_init),
            })
    ens = manager["ens"]
    memidx2run = len(ens.mem_list) + np.arange(num_new_mem, dtype=int)
    num_chunks = np.ones(num_new_mem, dtype=int)
    for mem in memidx2run:
        coldstart_info = ens.default_coldstart(spinup_interval, duration)
        coldstart_info["pert_seq"].loc[dict(time=spinup_interval)] = manager["rng"].integers(low=ens.model_params["seed_min"],high=ens.model_params["seed_max"])
        coldstart_info["init_cond"]["x"] = hist_dns["x"].sel(dict(time=spinup_interval)).load()
        ens.initialize_new_member(Lorenz96EnsembleMember, coldstart_info)
    ens.run_batch(memidx2run, num_chunks)
    pickle.dump(manager, open(join(sens_dir,"manager"),"wb"))
    return sens_dir

def run_wilks_parameterization(dns_dir, num_new_mem, seed_init, spinup_interval_phys, degree=4, persistence=None, overwrite_pmtzn=False):
    # Do a quartic fit to the tendencies and run stochastic parameterization
    ens_dns = pickle.load(open(join(dns_dir,"output","ens"),"rb"))
    ens_hist = ens_dns.load_member_ancestry(len(ens_dns.mem_list)-1)
    model_params = ens_dns.model_params.copy()
    pmtzn_dir = join(expt_dir,f"wilks_deg{degree}_pers{persistence}").replace(".","p")
    makedirs(pmtzn_dir, exist_ok=True)
    spinup_interval = int(spinup_interval_phys / model_params["time_unit"]) # after this interval, we branch other ensemble members
    restart_interval = ens_hist.time.size
    if overwrite_pmtzn or (not exists(join(pmtzn_dir, "pmtzn_wilks"))):
        Lorenz96Ensemble.fit_pmtzn_wilks(ens_dns, pmtzn_dir, spinup_interval, degree=degree, persistence=persistence, plot_flag=True)
    pmtzn = pickle.load(open(join(pmtzn_dir,"pmtzn_wilks"),"rb"))
    print(f"pmtzn = \n{pmtzn}")
    # Reduce the autocorrelation parameter by the chosen factor

    ensemble_size_limit = 1
    dirs_ens = dict({
        "work": join(pmtzn_dir,"work"),
        "output": join(pmtzn_dir,"output"),
        "home": ens_dns.dirs["home"],
        })

    # Modify the parameter dictionary for the parameterized version of the model
    model_params["noise"].update({
        "type": "red",
        "sites": np.arange(model_params["K"]),
        "magnitude_at_site": np.sqrt(pmtzn["res_var"])*np.arange(model_params["K"]),
        })
    model_params.update({
        "J": 0, 
        "polycoef": pmtzn["polycoef"],
        "persistence": pmtzn["persistence"],
        })

    # Create a simple 'manager' (just a dictionary)
    if exists(join(pmtzn_dir,"manager")):
        manager = pickle.load(open(join(pmtzn_dir,"manager"),"rb"))
    else:
        manager = dict({
            "ens": Lorenz96Ensemble.default_init(dirs_ens, model_params, ensemble_size_limit),
            "rng": default_rng(seed=seed_init),
            })
    ens = manager["ens"]

    memidx2run = len(ens.mem_list) + np.arange(num_new_mem, dtype=int)
    num_chunks = np.ones(num_new_mem, dtype=int)
    for mem in memidx2run:
        coldstart_info = ens.default_coldstart(spinup_interval, restart_interval-spinup_interval)
        coldstart_info["pert_seq"].loc[dict(time=spinup_interval)] = manager["rng"].integers(low=ens.model_params["seed_min"],high=ens.model_params["seed_max"])
        coldstart_info["init_cond"]["x"] = ens_hist["x"].sel(dict(time=spinup_interval)).load()
        ens.initialize_new_member(Lorenz96EnsembleMember, coldstart_info)
    ens.run_batch(memidx2run, num_chunks)
    pickle.dump(manager, open(join(pmtzn_dir,"manager"),"wb"))
    return pmtzn_dir

def visualize_several_integrations(ctrl_dir_list,savedir):
    # Stack a few Hovmuller diagrams together
    fig_hov,axes_hov = plt.subplots(
            nrows=len(ctrl_dir_list), ncols=2, figsize=(12,4*len(ctrl_dir_list)), 
            sharex=True, sharey=False, gridspec_kw={"hspace": 0.1, "wspace": 0.075}, layout="constrained")
    fig_x0,axes_x0 = plt.subplots(figsize=(10,4))
    titlesize = 22
    labelsize = 20
    ticksize = 18
    matplotlib.rcParams.update({'font.size': titlesize, 'axes.labelsize': labelsize})
    tspan = [500,515]
    handles_x0 = []
    for i_ctrl_dir,ctrl_dir in enumerate(ctrl_dir_list):
        ens_ctrl = pickle.load(open(join(ctrl_dir, "output", "ens"),"rb"))
        par = ens_ctrl.model_params
        sig = par["noise"]["magnitude_at_wavenumber"][0]
        #label = r"$a=%g$, $F_4=%g$"%(par['a'],sig)
        label = r"$F_4=%g$"%(sig)
        tu = ens_ctrl.model_params["time_unit"]
        dns = ens_ctrl.mem_list[0].load_history_selfmade().sel(time=slice(int(tspan[0]/tu), int(tspan[1]/tu))).load()
        print(f"dns['x'].shape = {dns['x'].shape}")
        tphys = dns["time"] * tu
        dns = dns.assign_coords(time=tphys)
        K = ens_ctrl.model_params["K"]
        krollfun = lambda h: h.roll(k=K//2).assign_coords(k=(h["k"] - K*(h["k"] >= K//2)).roll(k=K//2))

        ax = axes_hov[i_ctrl_dir,1]
        xr.plot.pcolormesh(krollfun(dns["x"]), x="time", y="k", cmap="BrBG", ax=ax, cbar_kwargs={"orientation": "vertical", "shrink": 1.0, "pad": 0.0025, "aspect": 15, 'label': ''})
        #ax.collections[0].colorbar.set_label(r"$x_k$", rotation=0, loc='center', verticalalignment='center', labelpad=0.1)
        cbar_ax = ax.collections[0].colorbar.ax
        cbar_ax.text(0.5, 1.02, r"$x_k$", transform=cbar_ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontsize=labelsize)
        ax.set_xlabel("")
        ax.set_ylabel("Longitude k", fontsize=labelsize)
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        ax.set_title(label)

        ax = axes_hov[i_ctrl_dir,0]
        h, = xr.plot.plot(dns["x"].isel(k=0).sel(time=slice(500,525)), ax=ax, color="black", linestyle="-")
        ax.set_xlabel("")
        ax.set_ylabel(r"$x_0$", fontsize=labelsize)
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        ax.set_title(label)

        ax = axes_x0
        linestyle = '--' if par['a']==0 else '-'
        h, = xr.plot.plot(dns["x"].isel(k=0).sel(time=slice(500,525)), ax=ax, color=plt.cm.rainbow(np.log1p(sig)), linestyle=linestyle, label=label)
        handles_x0.append(h)
    axes_hov[-1,0].set_xlabel("Time", fontsize=labelsize)
    axes_hov[-1,1].set_xlabel("Time", fontsize=labelsize)
    for ax in axes_hov.flat:
        ax.tick_params(axis='both', which='both', labelsize=ticksize)

    axes_x0.set_xlabel("Time")
    axes_x0.set_ylabel(r"$x_0$")
    axes_x0.set_title("")
    axes_x0.legend(handles=handles_x0,loc=(-0.42,0.3))
    fig_hov.savefig(join(savedir,f"hovmuller_multiparam"),**svkwargs)
    fig_x0.savefig(join(savedir,f"x0_multiparam"),**svkwargs)
    plt.close(fig_hov)
    plt.close(fig_x0)
    return
        


def visualize_long_integration(ctrl_dir):
    # Plot a slice of time surrounding the most extreme spike in energy at k=0
    # Select a timespan
    ens_ctrl = pickle.load(open(join(ctrl_dir, "output", "ens"),"rb"))
    par = ens_ctrl.model_params
    tu = ens_ctrl.model_params["time_unit"]
    #dns = ens_ctrl.load_member_ancestry(len(ens_ctrl.mem_list)-1).sel(time=slice(int(2/tu), None)).load()
    dns = ens_ctrl.mem_list[0].load_history_selfmade().sel(time=slice(int(50/tu), None)).load()
    print(f"dns['x'].shape = {dns['x'].shape}")
    tphys = dns["time"] * tu
    dns = dns.assign_coords(time=tphys)
    x0 = dns["x"].isel(k=0)
    x1 = dns["x"].isel(k=1)
    tidx_max = np.abs(x0).argmax("time").item()
    #tspan = slice(max(0,tidx_max-int(20/tu)), min(len(tphys)-1,tidx_max+int(20/tu)))
    tspan = slice(int(-50/tu), None)
    tspan_y = slice(int(-5/tu), None)

    # ---------- Figure for x ---------------
    vmax = 2.5*ens_ctrl.model_params["F"]

    fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(15,10),sharex="col",sharey="row",gridspec_kw={"width_ratios": [4,1], "height_ratios": [1,1]}, layout="constrained")
    fig.set_facecolor("white")
    config_label,config_display = Lorenz96Ensemble.label_from_config(par)
    axes[0,1].text(0.5, 0.5, config_display, horizontalalignment="center", verticalalignment="center", transform=axes[0,1].transAxes)

    # Upper left: Hovmuller diagram
    K = ens_ctrl.model_params["K"]
    krollfun = lambda h: h.roll(k=K//2).assign_coords(k=(h["k"] - K*(h["k"] >= K//2)).roll(k=K//2))
    ax = axes[0,0]
    xr.plot.pcolormesh(krollfun(dns["x"].isel(time=tspan)), y="k", cmap="BrBG", ax=ax, cbar_kwargs={"orientation": "vertical", "shrink": 0.6, "pad": 0.0025})
    ax.set_xlabel("Time")
    ax.set_ylabel("Longitude k")
    ax.xaxis.set_tick_params(which="both",labelbottom=True)
    # Lower left: timeseries
    ax = axes[1,0]
    h0, = xr.plot.plot(dns["x"].isel(time=tspan,k=0), x="time", ax=ax, label=r"$x(k=0)$", color="red")
    h1, = xr.plot.plot(dns["x"].isel(time=tspan,k=1), x="time", ax=ax, label=r"$x(k=1)$", color="dodgerblue")
    ax.set_ylim([-vmax,vmax])
    ax.legend(handles=[h0,h1])
    ax.set_title("")
    ax.set_xlabel("Time")
    # Lower right; histogram
    ax = axes[1,1]
    bin_edges = np.linspace(x0.min()-1e-10, x0.max()+1e-10, 20)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    hist0,_ = np.histogram(x0.to_numpy(), bins=bin_edges, density=True)
    hist1,_ = np.histogram(x1.to_numpy(), bins=bin_edges, density=True)
    histall,_ = np.histogram(dns["x"].to_numpy().flatten(), bins=bin_edges,density=True)
    h0, = ax.plot(hist0,bin_centers,label="$x(k=0)$",color="red",)
    h1, = ax.plot(hist1,bin_centers,label="$x(k=1)$",color="dodgerblue")
    h2, = ax.plot(histall,bin_centers,label="$x(k=$all$)$",color="black")
    #ax.legend(handles=[h0,h1,h2],loc=(0,1))
    ax.set_xlim([np.sort(histall)[1],np.max(histall)])
    ax.set_ylim([-vmax,vmax])
    ax.set_xscale("log")
    ax.set_xlabel("Probability density")
    ax.yaxis.set_tick_params(which="both",labelbottom=True)

    axes[0,1].axis("off")

    t0,t1 = (dns.time.isel(time=tspan).to_numpy()[[0,-1]]/tu).astype(int)
    print(f"{t0 = }, {t1 = }")
    fig.savefig(join(ctrl_dir,f"dns_x_{t0}-{t1}"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # -------------- Figure for y ---------------
    if par["J"] > 0:
        y0 = dns["y"].isel(k=0,j=0)
        y1 = dns["y"].isel(k=0,j=1)
        fig,axes = plt.subplots(ncols=2,nrows=2,figsize=(15,15),sharex="col",gridspec_kw={"width_ratios": [2,1], "height_ratios": [1,1]})
        fig.set_facecolor("white")
        fig.suptitle(r"$K=%d$, $J=%d$, $F=%.1f$"%(par['K'],par['J'],par['F']))

        # Upper left: Hovmuller diagram
        ax = axes[0,0]
        yst = dns["y"].isel(time=tspan_y,k=[0,1]).stack(kj=("k","j"))
        yst = yst.assign_coords(kj=np.arange(yst["kj"].size))
        print(f"yst.dims = {yst.dims}")
        xr.plot.pcolormesh(yst, y="kj", cmap="coolwarm", ax=ax, cbar_kwargs={"orientation": "horizontal"})
        ax.set_xlabel("Time")
        ax.set_ylabel("Longitude kj")
        ax.set_title("Hovmuller")
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        # Lower left: timeseries
        ax = axes[1,0]
        h0, = xr.plot.plot(dns["y"].isel(time=tspan,k=0,j=0), x="time", ax=ax, label=r"$y(k=0,j=0)$", color="red")
        h1, = xr.plot.plot(dns["y"].isel(time=tspan,k=1,j=1), x="time", ax=ax, label=r"$y(k=0,j=1)$", color="dodgerblue")
        ax.legend(handles=[h0,h1])
        ax.set_title("")
        # Lower right; histogram
        ax = axes[1,1]
        bin_edges = np.linspace(y0.min()-1e-10, y0.max()+1e-10, 20)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        hist0,_ = np.histogram(y0.to_numpy(), bins=bin_edges, density=True)
        hist1,_ = np.histogram(y1.to_numpy(), bins=bin_edges, density=True)
        histall,_ = np.histogram(dns["y"].to_numpy().flatten(), bins=bin_edges,density=True)
        h0, = ax.plot(hist0,bin_centers,label="$y(k=0,j=1)$",color="red",)
        h1, = ax.plot(hist1,bin_centers,label="$y(k=0,j=1)$",color="dodgerblue")
        h2, = ax.plot(histall,bin_centers,label="$y(k,j=$all$)$",color="black")
        ax.legend(handles=[h0,h1,h2])
        ax.set_xlim([np.sort(histall)[1],np.max(histall)])
        ax.set_xscale("log")
        ax.set_xlabel("Frequency")

        fig.savefig(join(ctrl_dir,f"dns_y_{tspan.start}-{tspan.stop}"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    return

def visualize_stochastic_pmtzn_ensemble(sens_dir, dns_dir):
    manager = pickle.load(open(join(sens_dir,"manager"),"rb"))
    ens_ctrl = pickle.load(open(join(dns_dir,"output","ens"),"rb"))
    hist_ctrl = ens_ctrl.load_member_ancestry(len(ens_ctrl.mem_list)-1).load()
    hist_ens = xr.concat([manager["ens"].load_member_ancestry(i_mem) for i_mem in range(len(manager["ens"].mem_list))], dim="member")
    tu = ens_ctrl.model_params["time_unit"]
    tmax = int(30/tu)
    t0 = hist_ens.time[0].item()

    fig = plt.figure(constrained_layout=True, figsize=(15,15))
    gs = GridSpec(3,2,figure=fig)

    # Top right: parameter specification
    ax01 = fig.add_subplot(gs[0,1])
    _,paramdisp = Lorenz96Ensemble.label_from_config(manager["ens"].model_params) 
    ax01.text(0.5, 0.5, paramdisp, horizontalalignment="left", verticalalignment="center")
    ax01.axis("off")

    # Top left: Hovmuller diagram for control
    ax00 = fig.add_subplot(gs[0,0])
    ax = ax00
    xr.plot.pcolormesh(hist_ctrl["x"].sel(time=slice(None,tmax)), x="time", y="k", cmap=plt.cm.coolwarm, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("k")
    ax.set_title("Control")

    # Middle left: mean of ensemble
    ax10 = fig.add_subplot(gs[1,0], sharex=ax00)
    ax = ax10
    xr.plot.pcolormesh(hist_ens["x"].sel(time=slice(None,tmax)).mean(dim="member"), x="time", y="k", cmap=plt.cm.magma, ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("k")
    ax.set_title("Ensemble mean")

    # Middle right: ensemble RMSE
    rmse = ((hist_ens["x"].sel(time=slice(None,tmax)) - hist_ctrl["x"].sel(time=slice(t0,tmax)))**2).mean(dim=["member","k"])
    ax11 = fig.add_subplot(gs[1,1], sharex=ax00)
    ax = ax11
    xr.plot.plot(rmse, x="time", ax=ax, color="red")
    ax.set_yscale("log")
    ax.set_title("Ensemble RMSE")

    # Lower left: spaghetti
    ax20 = fig.add_subplot(gs[2,0], sharex=ax00)
    ax = ax20
    for mem in hist_ens.member:
        xr.plot.plot(hist_ens["x"].sel(time=slice(None,tmax)).sel(member=mem,k=0), x="time", color="red", linewidth=1, ax=ax)
    xr.plot.plot(hist_ctrl["x"].sel(time=slice(None,tmax)).sel(k=0), x="time", color="black", linewidth=2, linestyle="--", ax=ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("x(k=0)")
    ax.set_title("One-site spread")

    # Align to align x axes
    #pos0 = ax00.get_position()
    #pos = ax.get_position()
    #ax.set_position([pos0.x0,pos.y0,pos0.width,pos.height])
    #ax.axvline(x=t0, color="red")

    # Lower right: histograms of control and parameterization
    ax21 = fig.add_subplot(gs[2,1], sharey=ax20)
    ax = ax21
    bin_edges = np.linspace(min(hist_ctrl["x"].min().compute().item(),hist_ens["x"].min().compute().item())-1e-10,max(hist_ctrl["x"].max().compute().item(),hist_ens["x"].max().compute().item())+1e-10,20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
    pdf_ens,_ = np.histogram(hist_ens["x"].sel(time=slice(t0,None)).to_numpy().flatten(), bins=bin_edges, density=True)
    pdf_ctrl,_ = np.histogram(hist_ctrl["x"].sel(time=slice(t0,None)).to_numpy().flatten(), bins=bin_edges, density=True)
    hens, = ax.plot(pdf_ens, bin_centers, color="red", marker=".", label="Stochastic")
    hctrl, = ax.plot(pdf_ctrl, bin_centers, color="black", marker=".", label="Control")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency")
    ax.legend(handles=[hens,hctrl])


    #for (i,j) in zip([1,2,1],[0,0,1]):
    #    fig.axes[i,j].get_shared_x_axes().join(axes[0,0], axes[i,j])
    fig.savefig(join(sens_dir,"plot_ensemble"), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return


def main_dns():

    tododict = dict({
        "run_dns":              1,
        "visualize_dns":        1,
        })

    loop_through_sigmas = False
    params_from_sysargs = True
    num_repetitions = 4
    siglist = np.array([3.0,3.0,1.0,0.5,0.25,0.0])
    alist = np.array([0.0] + [1.0]*5)

    ensemble_size_limit = 1
    computer = "engaging"
    if computer == "engaging":
        home_dir = "/home/ju26596/rare_event_simulation/TEAMS_L96"
        scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96"
    date_str = "2024-01-31"
    sub_date_str = "0"
    # TODO enable chaining together different files
    duration_dns_phys = 20000
    split_interval_phys = 2*duration_dns_phys # Not doing multiple perturbations for the control
    dns_supdir = join(scratch_dir, date_str, sub_date_str)

    config_dns_file = "./config_onetier.yml"  # Or a different file generated by a parameterization procedure
    config_model_dns = yaml.safe_load(open(config_dns_file,"r"))

    # Loop through the parameters
    if loop_through_sigmas:
        sigmas2run = siglist
        as2run = a_list
    elif params_from_sysargs:
        sigmas2run = [siglist[int(sys.argv[1])]]
        as2run = [alist[int(sys.argv[1])]]
    else:
        sigmas2run = [config_model_dns["noise"]["magnitude_at_wavenumber"][0]]
        as2run = [config_model_dns['a']]
    for rep in range(num_repetitions):
        for (a,sigma) in zip(as2run,sigmas2run):
            config_model_dns["noise"]["magnitude_at_wavenumber"][0] = sigma
            config_model_dns['a'] = a

            config_label,_ = Lorenz96Ensemble.label_from_config(config_model_dns)

            dns_dir = join(dns_supdir, config_label, "DNS")
            
            # run DNS to generate learning data for the pmtzn
            if tododict["run_dns"]:
                run_control(home_dir, dns_dir, config_model_dns, duration_dns_phys, split_interval_phys)
            if tododict["visualize_dns"]:
                visualize_long_integration(dns_dir)
    return

def dns_meta_analysis():
    print(f'Starting meta-analysis')
    # compare PDFs (especially tails) of DNS at different noise levels
    tododict = dict({
        "plot_hovmuller": 0,
        "compute_stats": 1,
        "plot_stats": 1,
        })
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_L96_results/examples/lorenz96"
    date_str = "2024-01-31"
    sub_date_str = "0"
    siglist = np.array([3.0,3.0,1.0,0.5,0.25,0.0])
    alist = np.array([0.0] + [1.0]*5)
    T_phys_list = np.array([30.0,15.0,6.0,4.5])[[2,]]
    resolution_list = np.array([1,])
    # TODO enable chaining together different files
    expt_dir = join(scratch_dir, date_str, sub_date_str)
    first_forcing_dir = join(expt_dir,(f"F6p0_K40_J0_a{alist[0]}_white_wave4mag{siglist[0]}").replace(".","p"))
    first_ens = pickle.load(open(join(first_forcing_dir,"DNS","output","ens"), "rb"))
    tu = first_ens.model_params["time_unit"]
    meta_dir = join(expt_dir,"meta_dns")
    makedirs(meta_dir, exist_ok=True)
    colors = plt.cm.coolwarm(np.arange(len(siglist))[::-1]/(len(siglist)-1))
    colors[0] = colors[1] # ame value for both 3.0s
    linestyles = ["--"] + 5*["-"]

    if tododict["plot_hovmuller"]:
        ctrl_dir_list = [join(expt_dir,(f"F6p0_K40_J0_a{alist[i]}_white_wave4mag{siglist[i]}").replace(".","p"),"DNS") for i in [5,2,1]]
        visualize_several_integrations(ctrl_dir_list,meta_dir)


    if tododict["compute_stats"]:

        hist_x = dict()
        binc_x = dict()
        hist_xsq = dict()
        binc_xsq = dict()
        rlev = dict()
        rlsf = dict()
        rtbf = dict() # expected return time from brute force
        rtthe = dict()

        for i_sig,(sig,a) in enumerate(zip(siglist,alist)):
            forcing_dir = join(expt_dir,(f"F6p0_K40_J0_a{a}_white_wave4mag{sig}").replace(".","p"))
            ens = pickle.load(open(join(forcing_dir,"DNS","output","ens"), "rb"))
            n_mem = len(ens.mem_list)
            tu = ens.model_params["time_unit"]
            x = ens.load_member_ancestry(n_mem-1)["x"].sel(k=0,time=slice(50/tu,(50+3.2e7)/tu)).load()
            xsq = x**2

            # Histogram the scores
            hist_x[i_sig],bin_edges = np.histogram(x.to_numpy(),density=True)
            binc_x[i_sig] = (bin_edges[:-1] + bin_edges[1:])/2
            hist_xsq[i_sig],bin_edges = np.histogram(xsq.to_numpy(),density=True)
            binc_xsq[i_sig] = (bin_edges[:-1] + bin_edges[1:])/2

            # Estimate return periods by maybe theory
            if a == 0:
                rtthe[i_sig] = ens.calculate_return_time_theoretical(binc_xsq[i_sig])
            
            # Estimate return periods via MBM and brute force
            rlev[i_sig] = dict()
            rlsf[i_sig] = dict()
            rtbf[i_sig] = dict() 
            for iT,T_phys in enumerate(T_phys_list):
                rlev[i_sig][iT] = dict()
                rlsf[i_sig][iT] = dict()
                for res in resolution_list:
                    T = int(round(T_phys / (res*tu)))
                    xsq_downsampled = xsq.isel(time=np.arange(0,xsq.time.size,res))
                    boot_sample_size = int(xsq_downsampled.time.size/T)
                    rlev[i_sig][iT][res],rlsf[i_sig][iT][res],bm,gevpar = utils.estimate_return_level_mbm(xsq_downsampled, T, 0, boot_sample_size_list=[boot_sample_size], n_boot=5000)
                    if iT == 0:
                        rtbf[i_sig][res] = xr.DataArray(coords={"lev": binc_xsq[i_sig]}, dims=['lev'], data=res*tu*utils.estimate_expected_hitting_time(xsq_downsampled.to_numpy(), binc_xsq[i_sig]))
        pickle.dump(hist_x,open(join(meta_dir,"hist_x"),"wb"))
        pickle.dump(hist_xsq,open(join(meta_dir,"hist_xsq"),"wb"))
        pickle.dump(binc_x,open(join(meta_dir,"binc_x"),"wb"))
        pickle.dump(binc_xsq,open(join(meta_dir,"binc_xsq"),"wb"))
        pickle.dump(rlev,open(join(meta_dir,"rlev"),"wb"))
        pickle.dump(rlsf,open(join(meta_dir,"rlsf"),"wb"))
        pickle.dump(rtbf,open(join(meta_dir,"rtbf"),"wb"))
        pickle.dump(rtthe,open(join(meta_dir,"rtthe"),"wb"))
    hist_x = pickle.load(open(join(meta_dir,"hist_x"),"rb"))
    hist_xsq = pickle.load(open(join(meta_dir,"hist_xsq"),"rb"))
    binc_x = pickle.load(open(join(meta_dir,"binc_x"),"rb"))
    binc_xsq = pickle.load(open(join(meta_dir,"binc_xsq"),"rb"))
    rlev = pickle.load(open(join(meta_dir,"rlev"),"rb"))
    rlsf = pickle.load(open(join(meta_dir,"rlsf"),"rb"))
    rtbf = pickle.load(open(join(meta_dir,"rtbf"),"rb"))
    rtthe = pickle.load(open(join(meta_dir,"rtthe"),"rb"))

    if tododict["plot_stats"]:
        # Return period curves
        # TODO plot for different downsamplings
        for iT,T_phys in enumerate(T_phys_list):
            for i_res,res in enumerate(resolution_list):
                fig,axes = plt.subplots(ncols=2,figsize=(16,5),gridspec_kw={"wspace": 0.3})

                ax = axes[0]
                handles = []
                ylim = [-np.inf,-np.inf]
                for i_sig,sig in enumerate(siglist):
                    if alist[i_sig] > 0:
                        label = r"$a=%g,F_4=%g$"%(alist[i_sig],sig)
                        label = r"$F_4=%g$"%(sig)
                        h, = ax.plot(binc_x[i_sig],hist_x[i_sig],color=colors[i_sig],linestyle=linestyles[i_sig],linewidth=4,label=label)
                        ylim[0] = max(ylim[0], max(hist_x[i_sig][[0,-1]]))
                        ylim[1] = max(ylim[1], np.max(hist_x[i_sig]))
                        handles.append(h)
                ax.set_ylim(ylim)
                ax.set_xlim([-12,16])
                ax.set_yscale("log")
                ax.set_ylabel("Probability density")
                ax.set_xlabel(r"$x_0$")
                ax.yaxis.set_tick_params(which="both",labelbottom=True)
                ax.set_title("")
                ax.legend(handles=handles,loc=(-0.5,0.25),edgecolor="white")

                ax = axes[1]
                for i_sig,(sig,a) in enumerate(zip(siglist,alist)):
                    if alist[i_sig] > 0:
                        T_phys = T_phys_list[iT]
                        T = int(round(T_phys / tu))
                        lsf2rt_coords = lambda rrr: rrr.assign_coords(lsf=-T*tu/np.log(-np.expm1(rrr["lsf"]))).rename(lsf="rt")
                        rlev[i_sig][iT][res] = lsf2rt_coords(rlev[i_sig][iT][res])
                        #rtime = -T/np.log(-np.expm1(rlsf[i_sig][iT]))
                        h, = xr.plot.plot(rlev[i_sig][iT][res].sel(est="empirical",confint=0,side="lo").isel(bss=0), x="rt", ax=ax, color=colors[i_sig], label=r"$F_4=%g,a=%g$, downsample $=%d$"%(sig,alist[i_sig],res), linewidth=4, linestyle=linestyles[i_sig])
                        ax.fill_between(
                                rlev[i_sig][iT][res]["rt"].to_numpy(),
                                2*rlev[i_sig][iT][res].sel(est="empirical",confint=0,side="lo").isel(bss=0) - rlev[i_sig][iT][res].sel(est="empirical",confint=0.95,side="lo").isel(bss=0),
                                2*rlev[i_sig][iT][res].sel(est="empirical",confint=0,side="lo").isel(bss=0) - rlev[i_sig][iT][res].sel(est="empirical",confint=0.95,side="hi").isel(bss=0),
                                facecolor=colors[i_sig],edgecolor="none",zorder=-1,alpha=0.3)

                        if False and a == 0:
                            # theoretical return period
                            thresh_list_x2 = rlsf[i_sig][iT][res].lev.to_numpy()
                            hthe, = ax.plot(rtthe[i_sig].values, rtthe[i_sig].level, linestyle="dotted", color="black", label=r"$F_4=%.2f,a=%.1f$"%(sig,alist[i_sig]))
                            hbf, = ax.plot(rtbf[i_sig][res].values, rtbf[i_sig][res].lev, color="dodgerblue", marker="o")
                        

                ax.set_xscale("log")
                ax.xaxis.set_tick_params(which="both",labelbottom=True)
                ax.yaxis.set_tick_params(which="both",labelbottom=True)
                ax.set_xlabel("Return period")
                ax.set_ylabel(r"Return level $(x_0^2)$")
                ax.set_title("")

                fig.savefig(join(meta_dir,(f"rlev_T{T_phys}_res{res}").replace(".","p")),**svkwargs)
                plt.close(fig)

        fig,axes = plt.subplots(ncols=2, figsize=(12,6))
        ax = axes[0]
        handles = []
        for i_sig,sig in enumerate(siglist):
            h, = ax.plot(binc_x[i_sig],hist_x[i_sig],color=colors[i_sig],linestyle=linestyles[i_sig],marker="o",label=r"$F_4=%g,a=%g$"%(sig,alist[i_sig]))
            handles.append(h)
        ax.set_yscale("log")
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"Probability density")
        ax = axes[1]
        for i_sig,sig in enumerate(siglist):
            ax.plot(binc_xsq[i_sig],hist_xsq[i_sig],color=colors[i_sig],linestyle=linestyles[i_sig],marker="o",label=r"$F_4=%g$"%(sig))
        ax.set_yscale("log")
        ax.set_xlabel(r"$x_0^2$")
        ax.set_ylabel("Probability density")
        ax.legend(handles=handles, loc="upper right")
        fig.savefig(join(meta_dir,"xhist"),**svkwargs)
        plt.close(fig)

    return

    
if __name__ == "__main__":
    main_dns()
    #dns_meta_analysis()

