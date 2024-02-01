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
from scipy.special import logsumexp,softmax
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

class TEAMSManager(ABC):
    def __init__(self, dirs, algo_params, init_pool):
        self.dirs = dirs.copy()
        for dirkey in list(self.dirs.keys()):
            os.makedirs(self.dirs[dirkey], exist_ok=False)
        self.algo_params = algo_params.copy() # time_units_per_chunk, 
        self.init_pool = init_pool # A queue of tuples of the form (file, time)



        # Initialize the books, including a triply nested dictionary of weights.
        self.scores_single = []
        self.scores_multiple = []
        #self.num_splits = [] # How many new children are birthed at each point along the trajectory
        self.max_scores = np.zeros(0)
        self.max_score_tidx = np.zeros(0, dtype=int)
        self.time_origins = np.zeros(0, dtype=int)
        self.times2split = np.zeros(0, dtype=int)
        self.headstarts = np.zeros(0, dtype=int) # Only applicable for descendants, not ancestors
        self.goals = np.zeros(0, dtype=float) # Only applicable for descendants
        self.max_splits_per_horizon = int(self.algo_params["time_horizon"]/self.algo_params["split_interval"]) - 1
        print(f"msph = {self.max_splits_per_horizon}")
        # For the random number generators, use them to generate seeds also 
        self.rng_resamp = default_rng(self.algo_params["seeddict"]["resample"])
        self.rng_perturb_plant = default_rng(self.algo_params["seeddict"]["perturb_plant"]) 
        self.rng_perturb_branch = default_rng(self.algo_params["seeddict"]["perturb_branch"]) 
        return 

    def set_level_raising_schedule(self, init_scores):
        sched = self.algo_params["politics"]["level_raising_schedule"]
        if sched["type"] == "const_frac2drop":
            params = dict(speed=sched["speed"])
        elif sched["type"] == "const_num2drop":
            params = dict(speed=sched["speed"])
        print(f"Finished setting the level-raising schedule: {params = }")
        return params



    def initialize_acq_state_global(self):
        asg = dict({
            "next_action": "plant",
            "previous_action": "",
            "level_schedule_params": dict(),
            "levels": [-np.inf],
            })
        print(f" /////////\n///////// INITIALIZING \n ////////// \n /////////")
        asg["active_members"] = np.zeros(0, dtype=int)
        asg["log_weights"] = np.zeros(0)
        asg["parent_queue"] = deque() 
        asg["multiplicities"] = np.zeros(0, dtype=int)
        self.acq_state_global = asg
        return
    
    def initialize_acq_state_local(self,new_member):
        self.acq_state_local[new_member] = dict()
        return

    def update_acq_state_local(self,member): 
        parent = self.acq_state_global["next_parent"] 
        print(f" ENTERING UPDATE ACQ STATE LOCAL ON MEMBER {member} (parent {parent})")
        asl = self.acq_state_local[parent]
        goal = self.acq_state_global["levels"][-1]
        asl["tidx_target"] = np.where(self.scores_single[parent].to_numpy() > goal)[0][0] 
        astr = self.algo_params["advance_split_time_range"]
        if astr[1] == astr[0]:
            ast = astr[0]
        else:
            ast = self.rng_perturb_branch.integers(low=astr[0],high=astr[1])

        asl["tidx_split_next"] = min(self.scores_single[parent].size-1, max(0, asl["tidx_target"] - ast))
        asl["tidx_split_next"] -= asl["tidx_split_next"] % self.algo_params["split_interval"]
        asl["headstart_actual"] = asl["tidx_target"] - asl["tidx_split_next"]
        # The next pert_seq will mostly copy that of the current parent, but perturb in advance
        asl["t_split_next"] = asl["tidx_split_next"] + self.time_origins[parent]
        print(f'{asl["t_split_next"] = }')

        # Either introduce new perturbations at a single point in time, or at all times following the split 
        new_pert_times = np.sort(np.union1d(asl["t_split_next"], self.ens.mem_list[parent].pert_seq.time.to_numpy()))
        times2changepert = np.array([asl["t_split_next"]])
        if self.algo_params["acquisition"]["perturb_everywhen_TEAMS"]:
            new_pert_times = new_pert_times[new_pert_times <= asl["t_split_next"]]
        asl["pert_seq_next"] = xr.DataArray(
                coords={"time": new_pert_times,},
                dims=["time",],
                data=0
                )
        changed_pert = self.rng_perturb_branch.integers(low=self.ens.model_params["seed_min"],high=self.ens.model_params["seed_max"],size=(len(times2changepert)))

        # TODO accept or reject each changed perturbation:
        legacy_pert_times = np.intersect1d(new_pert_times,self.ens.mem_list[parent].pert_seq.time)
        asl["pert_seq_next"].loc[dict(time=legacy_pert_times)] = self.ens.mem_list[parent].pert_seq.sel(time=legacy_pert_times).to_numpy()
        asl["pert_seq_next"].loc[dict(time=times2changepert)] = changed_pert 
        self.acq_state_local[parent] = asl

        print(f" ----**************------")
        
        asl = self.acq_state_local[parent]
        asg = self.acq_state_global
        # Decide the time at which to split (pre-ancestral peak, for simplicity)
        asl["tidx_target"] = self.max_score_tidx[parent]
        asl["tidx_split_next"] = max(0, asl["tidx_target"] - self.algo_params["advance_split_time_range"][0])
        asl["tidx_split_next"] -= asl["tidx_split_next"] % self.algo_params["split_interval"]
        asl["headstart_actual"] = asl["tidx_target"] - asl["tidx_split_next"]
        asl["goal"] = self.acq_state_global["levels"][-1]
        # The next pert_seq will mostly copy that of the current parent, but perturb in advance
        asl["t_split_next"] = asl["tidx_split_next"] + self.time_origins[parent]
        print(f'{asl["t_split_next"] = }')


        return

    @abstractmethod
    def create_new_pert_seq(self, parent, tidx_split):
        """
        Generate a new perturbation sequence, splitting from the parent's at the given split time 
        """
        pass

    @abstractmethod
    def similarity(self, snap0, snap1):
        """
        Some notion of distance between two snapshots (not full time histories)
        """
        # Will maintain a similarity matrix for all the initial conditions --- and terminal conditions? 
        pass

    
    @abstractmethod
    def score_fun_multiple(self, score_multiple_ancestral):
        """
        Function to evaluate progress of a trajectory. Return an xarray.DataArray with a single coordinate, time, that matches the time of the hist_mem. 
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

    @abstractmethod
    def create_warmstart_from_file(self, init_file, init_time, pert_seq):
        pass
    @abstractmethod
    def create_warmstart_from_member(self, i_parent, pert_seq):
        pass

    def plant_tree(self, EnsMemClass):
        n_mem_init = 1 #self.algo_params["politics"]["batch_size"]
        memidx2run = len(self.ens.mem_list) + np.arange(n_mem_init)

        # Prepare the initial conditions
        for i_mem in memidx2run:
            init_file,init_time = self.init_pool.pop()
            #print(f"About to feed in init file \n{init_file}\n and init_time {init_time}")
            warmstart_info = self.create_warmstart_from_file(init_file, init_time) 
            print(f"warmstart_info[init_cond] = \n{warmstart_info['init_cond']}")
            self.ens.initialize_new_member(EnsMemClass, warmstart_info.copy())
            self.time_origins = np.concatenate((self.time_origins, [init_time]))
            self.times2split = np.concatenate((self.times2split, [init_time]))
            self.headstarts = np.concatenate((self.headstarts, [-9999]))
            self.goals = np.concatenate((self.goals, [-np.inf]))
            
            chunks_per_mem = self.algo_params["chunks_per_mem"] * np.ones(len(memidx2run), dtype=int)

        # Run the simulation
        self.ens.run_batch(memidx2run, chunks_per_mem, verbose=False)

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
        parent = self.acq_state_global["next_parent"]
        asl = self.acq_state_local[parent]
        pert_seq = asl["pert_seq_next"]
        t_split = asl["t_split_next"]
        #parent,pert_seq,t_split = self.acq_state_global["next_parent"],self.acq_state_global["next_pert_seq"],self.acq_state_global["next_t_split"]

        child = len(self.ens.mem_list)
        print(f"\n\n>>>>>>>>>> Chosen parent <<<<<<<<<<<<<<")
        print(f"lineage = {self.ens.address_book[parent]}")
        print(f"score family history = {self.max_scores[self.ens.address_book[parent]]}")
        print(f">>>>>>>>>>  <<<<<<<<<<<<<<\n\n")

        # Prepare the new trajectory
        warmstart_info = self.create_warmstart_from_member(parent, pert_seq) 
        self.ens.initialize_new_member(EnsMemClass, warmstart_info, parent)

        # Run the new trajectory
        memidx2run = [child]
        chunks_per_mem = self.algo_params["chunks_per_mem"] * np.ones(1, dtype=int)
        self.ens.run_batch(memidx2run, chunks_per_mem, verbose=False)

        # Record the child's score
        score_parent_mult = self.scores_multiple[parent]
        time_origin = self.ens.mem_list[parent].time_origin
        hist_child = self.ens.mem_list[child].load_history_selfmade()
        score_child_mult = self.score_fun_multiple(hist_child, score_parent_mult)
        score_child_sing = self.score_fun_single(score_child_mult)
        self.scores_multiple.append(score_child_mult) 
        self.scores_single.append(score_child_sing) 
        new_max_score = score_child_sing.max(dim="time").compute().item()
        self.max_scores = np.concatenate((self.max_scores, [new_max_score]))
        self.max_score_tidx = np.concatenate((self.max_score_tidx, [score_child_sing.argmax(dim="time").item()]))
        self.time_origins = np.concatenate((self.time_origins, [time_origin]))
        self.times2split = np.concatenate((self.times2split, [t_split]))
        self.headstarts = np.concatenate((self.headstarts, [asl["headstart_actual"]]))
        self.goals = np.concatenate((self.goals, [self.acq_state_global["levels"][-1]]))


        print(f"{len(self.ens.mem_list) = }")

        return

    def take_next_step(self, EnsMemClass):
        if self.acq_state_global["next_action"] == "terminate":
            print(f"Terminating")
        else:
            if self.acq_state_global["next_action"] == "plant":
                self.plant_tree(EnsMemClass)
                self.acq_state_global["previous_action"] = "plant"

            elif self.acq_state_global["next_action"] == "branch":
                #if self.acq_state_global["previous_action"] == "plant":
                #    self.set_level_raising_schedule(self.max_scores)
                print(f"About to branch")
                print(f"about to branch; population = {len(self.ens.mem_list)}")
                self.branch(EnsMemClass)
                print(f"finished branching; population = {len(self.ens.mem_list)}")
                self.acq_state_global["previous_action"] = "branch"

            # -------------- Determine the next action -------------------
            new_member = len(self.ens.mem_list)-1
            self.initialize_acq_state_local(new_member)
            self.update_acq_state_global(new_member)
            if self.acq_state_global["next_action"] == "branch":
                print(f"For the coming branch, {self.acq_state_global['next_parent'] = }")
                self.update_acq_state_local(new_member)
            self.save_state()
        if len(self.ens.mem_list) % 32 == 0:
            self.print_status()
        return

    def update_acq_state_global(self,new_member):
        print(f"Updating global acq state")
        n_mem = len(self.ens.mem_list)

        asg = self.acq_state_global
        all_ancestors = np.where([len(addr) == 1 for addr in self.ens.address_book])[0]
        descmat = self.ens.construct_descent_matrix()

        ancestor = self.ens.address_book[new_member][0]
        # 1. process new member
        if asg["previous_action"] == "plant":
            asg["active_members"] = np.concatenate((asg["active_members"], [new_member]))
            asg["log_weights"] = np.concatenate((asg["log_weights"], [0.0]))
            asg["multiplicities"] = np.concatenate((asg["multiplicities"], [1]))
            if sum(asg["multiplicities"]) == self.algo_params["politics"]["base_size"]:
                print(f"\n\n///////////////////// \nfinally hit base size!!! ////////////\n\n")
                asg["level_schedule_params"] = self.set_level_raising_schedule(self.max_scores)
                asg["next_action"] = "branch"
        else:
            parent = self.ens.address_book[new_member][-2]
            asg["surviving_log_weight"] = logsumexp(asg["log_weights"][asg["active_members"]], b=asg["multiplicities"][asg["active_members"]])
            print(f"surviving log weight = {asg['surviving_log_weight']}")
            # Adjust the weight of all active members to exactly preserve their total weight
            asg["log_weights"][asg["active_members"]] -= np.log(1 + np.exp(asg["log_weights"][parent] - asg["surviving_log_weight"]))
            asg["log_weights"] = np.concatenate((asg["log_weights"], [asg["log_weights"][parent]]))
            success_flag = 1*(self.max_scores[new_member] > asg["levels"][-1])
            if success_flag:
                print(f"CHILD SUCCESS")
                asg["parent_queue"].appendleft(new_member)
                asg["active_members"] = np.concatenate((asg["active_members"], [new_member]))
                asg["multiplicities"] = np.concatenate((asg["multiplicities"], [1]))
            else:
                print(f"CHILD FAILURE")
                asg["multiplicities"][parent] += 1
                asg["multiplicities"] = np.concatenate((asg["multiplicities"], [0]))
                asg["parent_queue"].appendleft(parent)

            
        # 2. decide how to raise the next members 
        summult = np.sum([mult for mult in asg["multiplicities"][asg["active_members"]]])
        print(f"{summult = }")
        if asg["next_action"] == "branch" and summult == self.algo_params["politics"]["base_size"]:
            # Raise the level
            sched = self.algo_params["politics"]["level_raising_schedule"]
            if sched["type"] == "const_num2drop":
                order = np.argsort(self.max_scores[asg["active_members"]])
                num2drop = sched["speed"]
                if num2drop == len(asg["active_members"]):
                    new_level = self.max_scores[asg["active_members"]][order[-1]] + 1e-10
                else:
                    new_level = 0.5 * (self.max_scores[asg["active_members"]][order[num2drop-1]] + self.max_scores[asg["active_members"]][order[num2drop]])
            elif sched["type"] == "const_frac2drop":
                order = np.argsort(self.max_scores[asg["active_members"]])
                num2drop = int(round(0.5 + len(asg["active_members"]) * sched["speed"]))
                if num2drop == len(asg["active_members"]):
                    new_level = self.max_scores[asg["active_members"]][order[-1]] + 1e-10
                else:
                    new_level = 0.5 * (self.max_scores[asg["active_members"]][order[num2drop-1]] + self.max_scores[asg["active_members"]][order[num2drop]])
            else:
                raise NotImplementedError()
            asg["levels"].append(new_level)
            # Eliminate any members who didn't make the cut
            asg["active_members"] = asg["active_members"][self.max_scores[asg["active_members"]] > new_level]
            # Further eliminate any members who are deemed to have maxed out their optimization
            if len(asg["active_members"]) == 0:
                asg["next_action"] = "terminate"
            ancestor_diversity = len(np.unique([self.ens.address_book[i_mem][0] for i_mem in asg["active_members"]]))
            print(f"DEI statement: {ancestor_diversity = }")
            if ancestor_diversity < self.algo_params["politics"]["min_ancestor_diversity"]:
                asg["next_action"] = "terminate"

            if asg["next_action"] != "terminate":
                # Refill the queue counting multiplicities
                asg["parent_queue"] = deque()
                mems2replicate = self.rng_resamp.permutation([mem for mem in asg["active_members"] for rep in range(asg["multiplicities"][mem])])
                for mem in mems2replicate:
                    asg["parent_queue"].appendleft(mem)

        if asg["next_action"] == "branch":
            asg["next_parent"] = asg["parent_queue"].pop()
        self.acq_state_global = asg


        print(f"After deciding what to do next, levels = \n{np.array2string(np.array(asg['levels']), precision=2)}")
        return




    def print_status(self):
        n_mem = len(self.ens.mem_list)
        ancestors = np.where([len(addr) == 1 for addr in self.ens.address_book])[0]
        A = self.ens.construct_descent_matrix()
        print(f" ! ! ! ! ! -------- Status report ------- ! ! ! ! ! ")
        print(f"\tlevel-raising schedule = \n\t{self.acq_state_global['level_schedule_params']}")
        print(f"\tNumber of members = {len(self.ens.mem_list)}")
        print(f"\tweights: min = {np.exp(np.min(self.acq_state_global['log_weights'])):.3e}, max = {np.exp(np.max(self.acq_state_global['log_weights'])):.3e}, sum = {np.exp(logsumexp(self.acq_state_global['log_weights'], b=self.acq_state_global['multiplicities'])):.3e} ")
        if len(self.max_scores) > 0:
            print(f"\t\t\t\t\t anc score -> (min score, max score, gain);    num desc")
            for anc in ancestors:
                idx_g = np.where(A[anc,:])[0]
                anc_score = self.max_scores[anc]
                min_score = np.min(self.max_scores[idx_g])
                max_score = np.max(self.max_scores[idx_g])
                print(f"\tAncestor {anc:03d}:  \t {anc_score:.2f} -> ({min_score:.2f},      {max_score:.2f},        {(max_score-anc_score):.2e});     {len(idx_g)} total desc ", end="")
                num_active_desc = np.sum(np.in1d(idx_g, self.acq_state_global["active_members"]))
                if num_active_desc > 0:
                    print(f"    ({num_active_desc} active)", end="")
                print(f"\n",end="")
        print(f"\n\tPrevious action: {self.acq_state_global['previous_action']}; next action: {self.acq_state_global['next_action']} \n")
        print(f" ! ! ! ! ! -------- End status report ------- ! ! ! ! ! ")
        return

    def save_state(self):
        filename = join(self.dirs["metadata"], f"manager")
        if exists(filename): 
            os.rename(filename, join(self.dirs["metadata"], "backup_manager"))
        pickle.dump(self, open(join(self.dirs["metadata"], f"manager"), "wb"))
        return

    @classmethod
    def plot_return_stats_multiple_leads(cls, rlev_list, T_list, tu, delta_list, delta_opt, savedir, suffix, label):
        fig,ax = plt.subplots()
        handles = []
        for i_param in range(len(rlev_list)):
            rlev = rlev_list[i_param].copy()
            rlev_teams = rlev["teams"]["split"]["sup"].sel(est="empirical",confint=0,side="lo")
            rlev_dns = rlev["dns"].sel(est="empirical",confint=0,side="lo").sel(bss=rlev["dns"].bss.max().item())
            rlev_teams = rlev_teams.assign_coords(lsf=-T_list[i_param]*tu/np.log(-np.expm1(rlev_teams["lsf"].to_numpy()))).rename(lsf="rt")
            rlev_dns = rlev_dns.assign_coords(lsf=-T_list[i_param]*tu/np.log(-np.expm1(rlev_dns["lsf"].to_numpy()))).rename(lsf="rt")
            linewidth = 4 if delta_opt == delta_list[i_param] else 0.75
            h, = xr.plot.plot(rlev_teams,x="rt",label=r"$\delta=$%.2f"%(delta_list[i_param]),color=plt.cm.viridis(i_param/len(rlev_list)), ax=ax, linewidth=linewidth)
            handles.append(h)
        h, = xr.plot.plot(rlev_dns,x="rt",linestyle="--",color="black",linewidth=2,label=r"DNS", ax=ax)
        handles.append(h)
        ax.legend(handles=handles,loc=(1,0))
        ax.set_title(label)
        ax.set_xscale("log")
        ax.set_xlabel("Return time")
        ax.set_ylabel("Return level")
        fig.savefig(join(savedir,f"rl_of_rt_multilead_{suffix}"),**svkwargs)
        return

        
    # Methods for analyzing output
    @classmethod 
    def estimate_return_statistics_multiple_runs(cls, mandict, F=None, bin_edges=None, lsf_interp=None, lev_interp=None):
        seeds = list(mandict.keys())
        if F is None: F = dict()
        logW = dict()
        logW_prog = dict()
        # Make separate collections just for the base
        F_base = dict()
        logW_base = dict()
        idx_base = dict()
        for seed in seeds:
            if seed not in F.keys():
                F[seed] = mandict[seed].max_scores 
            if "multiplicities" in mandict[seed].acq_state_global.keys():
                multiplicities = mandict[seed].acq_state_global["multiplicities"]
            else:
                multiplicities = np.ones(len(F[seed]), dtype=int)

            logW[seed] = mandict[seed].acq_state_global["log_weights"] + np.log(multiplicities) # Any zero multiplicities lead to negatively infinite weights
            # Base values
            idx_base[seed] = np.where([len(addr) == 1 for addr in mandict[seed].ens.address_book])[0]
            F_base[seed] = F[seed][idx_base[seed]]
            logW_base[seed] = np.zeros(len(idx_base[seed]))
        rlev = dict()
        rlsf = dict()
        gevpar = dict()
        rlev_boot = dict()
        rlsf_boot = dict()
        gevpar_boot = dict()
        gevpar_prog = dict()
        rlev["split"],rlsf["split"],gevpar["split"],rlev_boot["split"],rlsf_boot["split"],gevpar_boot["split"] = utils.estimate_return_statistics_many_ensembles(F, logW, lsf_interp=lsf_interp, lev_interp=lev_interp)
        lsf_interp = rlev["split"]["agg"]["lsf"].to_numpy()
        lev_interp = rlsf["split"]["agg"]["lev"].to_numpy()
        print(f"------------\n\tBEGINNING BASE ESTIMATE-----------------")
        print(f"lsf_interp = {lsf_interp}")
        rlev["init"],rlsf["init"],gevpar["init"],rlev_boot["init"],rlsf_boot["init"],gevpar_boot["init"] = utils.estimate_return_statistics_many_ensembles(F_base, logW_base, lsf_interp=lsf_interp, lev_interp=lev_interp)
        # TODO add another item in the dictionary for "init_rolled"
        for seed in seeds:
            Nlist_seed = np.linspace(len(idx_base[seed]), len(F[seed]), 10).astype(int)
            log_weights = mandict[seed].acq_state_global["log_weights"]
            logWlist_seed = [log_weights[:nls] for nls in Nlist_seed]
            gevpar_prog[seed] = utils.estimate_gev_params_progressively(F[seed], logWlist_seed, method="PWM")
        # Make a histogram
        score_bag = np.concatenate(tuple([F[seed] for seed in seeds]))
        score_bag_base = np.concatenate(tuple([F_base[seed] for seed in seeds]))
        if bin_edges is None:
            bin_edges = np.linspace(np.min(score_bag)-1e-10,np.max(score_bag)+1e-10,20)
        hist = dict()
        hist["split"],_ = np.histogram(score_bag, bins=bin_edges)
        hist["init"],_ = np.histogram(score_bag_base, bins=bin_edges)

        return rlev,rlsf,gevpar,rlev_boot,rlsf_boot,gevpar_boot,gevpar_prog,hist
        #return rlev_all,rlev_sep,rlev_all_base,rlev_sep_base,hist,hist_base,gevpar_all,gevpar_sep,gevpar_all_base,gevpar_sep_base

    @classmethod
    def plot_multiple_pooled_runs(cls, mandict_list, savefolder_list, metadir):
        for iman in range(len(mandict_list)):
            mandict = mandict_list[iman]
            savefolder = savefolder_list[iman]
            rlev = pickle.load(open(join(savefolder, "rlev"),"rb"))
            rlsf = pickle.load(open(join(savefolder, "rlsf"),"rb"))
            gevpar = pickle.load(open(join(savefolder, "gevpar"),"rb"))
            gevpar_prog = pickle.load(open(join(savefolder, "gevpar_prog"),"rb"))
            hist = pickle.load(open(join(savefolder, "hist"),"rb"))
            bin_edges = hist["dns_bin_edges"]
            if ylim is None:
                ylim = [bin_edges[0],1.0*bin_edges[-1]-0.0*bin_edges[0]]
            fig,ax = plt.subplots()
            # TODO
        return
    @classmethod
    def tabulate_performance_metrics(cls, mandict, savefolder, simtime_dns, score_dns, tododict, paramdisp=None, ylim=None, bootstrap_version="basic", plot_init_sep=True, plot_init_sup=True, plot_median=True, ):
        # Measure and save out the following
        # I. Comparisons with DNS
        #   A. Some integrated difference of survival function 
        # II. Diagnostics of algorithm's behavior, computable online
        #   A. Rejection rate and head start 
        seeds = list(mandict.keys())
        Thorz = mandict[seeds[0]].algo_params["time_horizon"]
        Twait = mandict[seeds[0]].algo_params["score"]["twait"]
        Tavg = mandict[seeds[0]].algo_params["score"]["tavg"]
        Tteams = Thorz - Twait - (Tavg - 1)
        Tdns = Thorz - Twait - (Tavg - 1)
        tu = mandict[seeds[0]].ens.model_params["time_unit"]
        N = mandict[seeds[0]].algo_params["politics"]["base_size"]
        n_mem_teams_sup = sum([len(mandict[seed].ens.mem_list) for seed in seeds])
        print(f"\n\n\t\t------------------------- {n_mem_teams_sup = } -----------------\n\n")
        n_mem_teams_sep = int(n_mem_teams_sup / len(seeds))
        n_blocks_dns = int(round(simtime_dns/Tdns)) # Gives the minimum exceedance probabilitywe might estimate with DNS

        # Define a standard range of tail probabilities at which to evaluate return levels
        max_level = np.max([np.max(mandict[seed].max_scores) for seed in seeds]) 
        print(f"max_level = {max_level}")

        # -------------------------------------------
        all_log_weights = np.concatenate(tuple([mandict[seed].acq_state_global["log_weights"] for seed in seeds]))
        fidx = np.where(np.isfinite(all_log_weights))[0]
        #lsf_min = np.min(all_log_weights[fidx]) - logsumexp(all_log_weights[fidx])
        lsf_min = -1.5*np.log(n_blocks_dns)
        print(f"lsf_min = {lsf_min}")
        lsf_interp = np.linspace(lsf_min, np.log(0.5), 30)[::-1]
        print(f"lsf_interp = {lsf_interp}")


        if tododict["tally_rejections"]:
            # Diagnose rejection vs. advance splitting time and goal 
            print(f"Diagnosing rejections")
            fig,axes = plt.subplots(ncols=len(seeds), nrows=3, figsize=(5*len(seeds),15))
            for i_seed,seed in enumerate(seeds):
                ax_col = axes if len(seeds)==1 else axes[:,i_seed]
                manager = mandict[seed]
                n_mem = len(manager.ens.mem_list)
                # First row: head start
                ax = ax_col[0]
                ax.plot(range(N,n_mem),tu*manager.headstarts[N:],color="black")
                ax.set_ylabel("Head start")
                # Second row: clearance
                ax = ax_col[1]
                ax.plot(range(N,n_mem),manager.max_scores[N:]-manager.goals[N:], color="black")
                ax.axhline(0, color="blue", linestyle="--", linewidth=3)
                ax.set_ylabel("Clearance")
                # Third row: scatter them
                ax = ax_col[2]
                ax.scatter(manager.headstarts[N:], manager.max_scores[N:]-manager.goals[N:], color="black", marker='.')
                ax.set_xlabel("Head start")
                ax.set_ylabel("Clearance")
            fig.savefig(join(savefolder,"rejection_diagnosis"),bbox_inches="tight",pad_inches=0.2)
            plt.close(fig)


        if tododict["summarize_tail"]:
            # ----------- Compute return levels and GEV parameters -----------
            rlev = dict()
            rlsf = dict()
            gevpar = dict()
            rlev_boot = dict()
            rlsf_boot = dict()
            gevpar_boot = dict()
            hist = dict()
            fdiv = dict() # A few different F-divergences 
            cost = dict()

            print(f"{Tteams = }, {Tdns = }")

            #lev_min = min(np.nanmin(score_dns), np.min([np.min(mandict[seed].max_scores) for seed in seeds]))
            #lev_max = max(np.nanmax(score_dns), np.max([np.max(mandict[seed].max_scores) for seed in seeds]))
            lev_min = np.nanquantile(score_dns.to_numpy(), 0.9)
            lev_max = 1.1*np.nanmax(score_dns).item() - 0.1*lev_min
            lev_interp = np.linspace(lev_min, lev_max, 40)
            print(f"lev_interp from dns = {lev_interp}")
            # 1. DNS
            boot_sample_size_list = (np.array([n_mem_teams_sup,n_mem_teams_sep]) * Thorz/Tdns).astype(int)
            print(f"{boot_sample_size_list = }")
            rlev["dns"],rlsf["dns"],dns_block_maxima,gevpar["dns"] = utils.estimate_return_level_mbm(score_dns, Tdns, Tavg-1, boot_sample_size_list, n_boot=5000, min_quantile=0.0,lsf_interp=lsf_interp, lev_interp=lev_interp)

            bin_edges = np.linspace(
                    -1e-10 + lev_min, #max(np.min(dns_block_maxima), min_level),
                    1e-10 + lev_max, #max(np.max(dns_block_maxima), max_level),
                    31
                    )
            hist["dns"],hist["dns_bin_edges"] = np.histogram(dns_block_maxima,bins=bin_edges)
            # 2. AMS
            func_vals_dict = dict({seed: mandict[seed].max_scores for seed in seeds})
            rlev["teams"],rlsf["teams"],gevpar["teams"],rlev_boot["teams"],rlsf_boot["teams"],gevpar_boot["teams"],gevpar_prog,hist["teams"] = cls.estimate_return_statistics_multiple_runs(mandict, F=func_vals_dict, bin_edges=bin_edges, lsf_interp=lsf_interp, lev_interp=lev_interp)
            cost["n_mem_teams"] = n_mem_teams_sup
            cost["n_blocks_dns"] = n_blocks_dns

            # Measure scalar differences between AMS and DNS tails
            pmf_teams = np.expm1(rlsf["teams"]["split"]["agg"].isel(lev=slice(1,None))) * np.exp(-rlsf["teams"]["split"]["agg"].diff("lev"))
            pmf_dns = np.expm1(rlsf["dns"].isel(bss=0,lev=slice(1,None))) * np.exp(-rlsf["dns"].diff("lev"))
            print(f"{pmf_teams = }")
            print(f"{pmf_dns = }")
            fdiv["chisq"] = ((pmf_dns - pmf_teams)**2 / pmf_dns).sum().item()
            fdiv["kl"] = (pmf_teams * np.log(pmf_teams / pmf_dns)).sum().item()
            fdiv["relerr"] = (((pmf_dns - pmf_teams)/pmf_dns)**2).sum().item()

            print(f"fdiv = \n{fdiv}")

            # ------- Save the results -----------
            pickle.dump(rlev, open(join(savefolder, "rlev"),"wb"))
            pickle.dump(rlsf, open(join(savefolder, "rlsf"),"wb"))
            pickle.dump(gevpar, open(join(savefolder, "gevpar"),"wb"))
            pickle.dump(rlev_boot, open(join(savefolder, "rlev_boot"),"wb"))
            pickle.dump(rlsf_boot, open(join(savefolder, "rlsf_boot"),"wb"))
            pickle.dump(gevpar_boot, open(join(savefolder, "gevpar_boot"),"wb"))
            pickle.dump(gevpar_prog, open(join(savefolder, "gevpar_prog"),"wb"))
            pickle.dump(hist, open(join(savefolder, "hist"),"wb"))
            pickle.dump(fdiv, open(join(savefolder, "fdiv"),"wb"))
            pickle.dump(cost, open(join(savefolder, "cost"),"wb"))
        rlev = pickle.load(open(join(savefolder, "rlev"),"rb"))
        rlsf = pickle.load(open(join(savefolder, "rlsf"),"rb"))
        gevpar = pickle.load(open(join(savefolder, "gevpar"),"rb"))
        gevpar_prog = pickle.load(open(join(savefolder, "gevpar_prog"),"rb"))
        hist = pickle.load(open(join(savefolder, "hist"),"rb"))
        bin_edges = hist["dns_bin_edges"]
        if ylim is None:
            elongation = 0.05
            ylim = [bin_edges[len(bin_edges)//4],(1+elongation)*bin_edges[-1]-elongation*bin_edges[len(bin_edges)//4]]

        Nmax = max([len(mandict[seed].ens.mem_list) for seed in seeds])

        # Convert CDFs to return periods
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


        if tododict["plot_return_stats"]:
            labelsize = 18
            titlesize = 24
            ticksize = 16
            simtime_teams = n_mem_teams_sup * Thorz
            simtime_teams_base = N * len(seeds) * Thorz 
            #simtime_dns = score_dns["time"].size * tu
            cost_disp = "\n".join([
                r"TEAMS cost:",
                r" %.1E"%(simtime_teams*tu/len(seeds)),
                r" $\times$ %d runs"%(len(seeds)),
                r" $=$%.1E"%(simtime_teams*tu),
                r" ",
                r"DNS cost: ",
                r" %.1E"%(simtime_dns*tu),
                ])

            seedstr = f"seeds{min(seeds)}to{max(seeds)}"

            # ------------- Return times as function of return level, and vice versa -----------------

            # Set limits for the plots

            xlim = [5*Thorz*tu, 5*simtime_dns*tu]

            print(f"rlsf['dns'] = {rlsf['dns']}")


            fig0,axes0 = plt.subplots(ncols=3, figsize=(18,5)) #, sharey=True)
            fig0.set_facecolor("white")
            fig1,axes1 = plt.subplots(ncols=3, figsize=(18,5)) #, sharey=True)
            fig1.set_facecolor("white")
            # -------- Averaging return level between different runs -------
            ax = axes0[0]
            handles00 = []
            for i_seed,seed in enumerate(rlsf["teams"]["split"]["sep"].seed):
                hteams_sep, = xr.plot.plot(rlev["teams"]["split"]["sep"].sel(seed=seed,est="empirical"), x="rt", color="red", linestyle="-", alpha=0.25, ax=ax, label=r"TEAMS")
                if i_seed == 0:
                    handles00.append(hteams_sep)
                if plot_init_sep:
                    hinit_sep, = xr.plot.plot(rlev["teams"]["init"]["sep"].sel(seed=seed,est="empirical"), x="rt", color="dodgerblue", linestyle="-", alpha=0.25, label=r"Init", ax=ax)
                    if i_seed == 0:
                        handles00.append(hinit_sep)
            if plot_median:
                hteams_median, = xr.plot.plot(rlev["teams"]["split"]["sep"].sel(est="empirical").quantile(0.5,dim="seed"), x="rt", color="red", linestyle="--", linewidth=3, ax=ax, label=f"TEAMS median")
                handles00.append(hteams_median)
                if plot_init_sep: 
                    hinit_median, = xr.plot.plot(rlev["teams"]["init"]["sep"].sel(est="empirical").quantile(0.5,dim="seed"), x="rt", color="dodgerblue", linestyle="--",linewidth=3, ax=ax, label="Init median")
                    handles00.append(hinit_median)
            hdns, = xr.plot.plot(rlev["dns"].isel(bss=0).sel(est="empirical",confint=0,side="lo"), x="rt", color="black", linestyle="-", label=r"DNS", ax=ax)
            handles00.append(hdns)

            if paramdisp is None: paramdisp = ""
            display = paramdisp + "\n\n" + cost_disp
            ax.text(-0.3,0.5,display,fontsize=20,transform=ax.transAxes,horizontalalignment="right",verticalalignment="center")
            ax.legend(handles=handles00,loc="lower right",fontsize=labelsize)
            ax.set_xlabel("Return period", fontsize=labelsize)
            ax.set_ylabel("Return level", fontsize=labelsize)
            ax.yaxis.set_tick_params(which="both",labelbottom=True,labelsize=ticksize)
            ax.set_title("Single TEAMS runs",fontsize=titlesize)
            ax.set_xscale("log")
            ax.set_xlim(xlim)
            #ax.set_ylim(ylim)

            def bootstrap_ci(rrr_sup):
                lower_quantile = rrr_sup.sel(est="empirical",confint=0.95,side="lo")
                upper_quantile = rrr_sup.sel(est="empirical",confint=0.95,side="hi")
                point_estimate = rrr_sup.sel(est="empirical",confint=0,side="lo")
                if bootstrap_version == "basic":
                    lower = 2*point_estimate - upper_quantile
                    upper = 2*point_estimate - lower_quantile
                elif bootstrap_version == "percentile":
                    lower = lower_quantile
                    upper = upper_quantile
                else:
                    raise Exception("{bootstrap_version = } but must be either basic or percentile")
                return point_estimate,lower,upper

            ax = axes0[1]
            handles01 = []
            point_teams,lower_teams,upper_teams = bootstrap_ci(rlev["teams"]['split']['sup'])
            point_init,lower_init,upper_init = bootstrap_ci(rlev["teams"]['init']['sup'])
            point_dns,lower_dns,upper_dns = bootstrap_ci(rlev["dns"].isel(bss=0))
            hteams_point, = xr.plot.plot(point_teams, x="rt", color="red", linestyle="-", alpha=1, linewidth=3, label=r"TEAMS", ax=ax)
            handles01.append(hteams_point)
            if plot_init_sup:
                hinit_point, = xr.plot.plot(point_init, x="rt", color="blue", linestyle="-", alpha=1, linewidth=3, label=r"Init", ax=ax)
                handles01.append(hinit_point)
            hdns_point, = xr.plot.plot(point_dns, x="rt", color="black", linestyle="-", label=r"DNS", ax=ax)
            handles01.append(hdns_point)
            # Error bars
            hteams_ci = ax.fill_between(
                    point_teams["rt"], lower_teams, upper_teams,
                    facecolor="red", edgecolor="none", alpha=0.3, 
                    label=r"TEAMS 95% CI")
            #handles01.append(hteams_ci)
            if plot_init_sup:
                hinit_ci = ax.fill_between(
                        point_init["rt"], lower_init, upper_init, 
                        facecolor="dodgerblue", edgecolor="none", alpha=0.3,
                        label="Init 95% CI")
                #handles01.append(hinit_ci)
            hdns_ci_sup = ax.fill_between(
                    point_dns["rt"], lower_dns, upper_dns,
                    facecolor="gray", edgecolor="none", alpha=0.5,
                    label=r"DNS 95% CI")
            #handles01.append(hdns_ci_sup)
            #ax.legend(handles=[hteams_point,hteams_ci,hinit_point,hinit_ci,hdns_point,hdns_ci],loc="upper right",bbox_to_anchor=(-1.6,1.2))
            ax.set_xlabel("Return period",fontsize=labelsize)
            ax.set_ylabel("")
            ax.yaxis.set_tick_params(which="both",labelbottom=True,labelsize=ticksize)
            ax.set_xscale("log")
            #ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.set_title("Pooled results", fontsize=titlesize)
            ax.legend(handles=handles01,loc="lower right",fontsize=labelsize)


            # --------------- Averaging horizontally ----------------
            ax = axes1[0]
            handles10 = []
            for i_seed,seed in enumerate(rlsf["teams"]["split"]["sep"].seed):
                hteams_sep, = xr.plot.plot(rlsf["teams"]["split"]["sep"].sel(seed=seed,est="empirical"), y="lev", color="red", linestyle="-", alpha=0.25, ax=ax, label=r"TEAMS single run")
                if i_seed == 0:
                    handles10.append(hteams_sep)
                if plot_init_sep:
                    hinit_sep, = xr.plot.plot(rlsf["teams"]["init"]["sep"].sel(seed=seed,est="empirical"), y="lev", color="dodgerblue", linestyle="-", alpha=0.25, label=r"Single init", ax=ax)
                    if i_seed == 0:
                        handles10.append(hinit_sep)
            hteams_mean, = ax.plot(rlsf_mean["split"], rlsf["teams"]["split"]["sep"]["lev"], color="red", linestyle="-", linewidth=3, label="TEAMS mean")
            handles10.append(hteams_mean)
            if plot_init_sep:
                hinit_mean, = ax.plot(rlsf_mean["init"], rlsf["teams"]["split"]["sep"]["lev"], color="dodgerblue", linestyle="-", linewidth=3, label="Init mean")
                handles10.append(hinit_mean)
            hdns, = xr.plot.plot(rlsf["dns"].isel(bss=0).sel(dict(est="empirical",confint=0,side="lo")), y="lev", color="black", linestyle="-", label=r"DNS", ax=ax)
            handles10.append(hdns)
            ax.legend(handles=handles10,loc="lower right",fontsize=labelsize)
            ax.set_xlabel("Return period", fontsize=labelsize)
            ax.set_ylabel("Return level", fontsize=labelsize)
            ax.set_title("Separate results", fontsize=titlesize)
            ax.set_xlim(xlim)
            #ax.set_ylim(ylim)
            ax.set_xscale("log")
            ax.yaxis.set_tick_params(which="both",labelbottom=True,labelsize=ticksize)
            ax.set_ylabel("")
            if paramdisp is not None:
                ax.text(-1.1,1.0,paramdisp,transform=ax.transAxes,horizontalalignment="left",verticalalignment="top")
            ax.text(-1.1,0.0,cost_disp,transform=ax.transAxes,horizontalalignment="left",verticalalignment="bottom")

            

            ax = axes1[1]
            handles11 = []
            point_teams,lower_teams,upper_teams = bootstrap_ci(rlsf["teams"]['split']['sup'])
            point_init,lower_init,upper_init = bootstrap_ci(rlsf["teams"]['init']['sup'])
            point_dns,lower_dns,upper_dns = bootstrap_ci(rlsf["dns"].isel(bss=0))
            hteams_point, = xr.plot.plot(point_teams, y="lev", color="red", linestyle="-", alpha=1, linewidth=3, label=r"TEAMS", ax=ax)
            handles11.append(hteams_point)
            if plot_init_sup:
                hinit_point, = xr.plot.plot(point_init, y="lev", color="blue", linestyle="-", alpha=1, linewidth=3, label=r"Init", ax=ax)
                handles11.append(hinit_point)
            hdns_point, = xr.plot.plot(point_dns, y="lev", color="black", linestyle="-", label=r"DNS", ax=ax)
            handles11.append(hdns_point)
            # Error bars
            hteams_ci = ax.fill_betweenx(
                    point_teams["lev"], lower_teams, upper_teams,
                    facecolor="red", edgecolor="none", alpha=0.3,
                    label=r"TEAMS 95% CI")
            handles11.append(hteams_ci)
            if plot_init_sup:
                hinit_ci = ax.fill_betweenx(
                        point_init["lev"], lower_init, upper_init,
                        facecolor="dodgerblue", edgecolor="none", alpha=0.3,
                        label="Init 95% CI")
                handles11.append(hinit_ci)
            hdns_ci_sup = ax.fill_betweenx(
                    point_dns["lev"], lower_dns, upper_dns,
                    facecolor="gray", edgecolor="none", alpha=0.5,
                    label="DNS 95% CI")
            handles11.append(hdns_ci_sup)
            ax.set_xlabel("Return period",fontsize=labelsize)
            ax.set_ylabel("Return level",fontsize=labelsize)
            ax.set_title("Pooled results",fontsize=titlesize)
            ax.legend(handles=handles11,loc="lower right",fontsize=labelsize)
            ax.set_xscale("log")
            #ax.set_ylim(ylim)
            ax.set_xlim(xlim)

            for ax in [axes0[2],axes1[2]]:
                bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
                ax.plot(hist["dns"], bin_centers, color="black", marker=".", label="DNS")
                ax.plot(hist["teams"]["init"], bin_centers, color="dodgerblue", marker=".", label="TEAMS init")
                ax.plot(hist["teams"]["split"], bin_centers, color="red", marker=".",)
                ax.set_xscale("log")
                ax.set_title("Score histogram",fontsize=titlesize)
                ax.set_xlabel("Counts",fontsize=labelsize)
                ax.yaxis.set_tick_params(which="both",labelbottom=True,labelsize=ticksize)
                #ax.set_ylim(ylim)

            # Make all y-limits adhere to those of the middle panel
            for axes in [axes0,axes1]:
                ylim = axes[1].get_ylim()
                axes[0].set_ylim(ylim)
                axes[2].set_ylim(ylim)

            fig0.savefig(join(savefolder,f"rl_of_rt_{seedstr}"), bbox_inches="tight", pad_inches=0.2)
            plt.close(fig0)
            fig1.savefig(join(savefolder,f"rt_of_rl_{seedstr}"), bbox_inches="tight", pad_inches=0.2)
            plt.close(fig1)
            print(f'Just saved two figs to {savefolder}. {simtime_dns*tu = }')


        if tododict["plot_gev_est"]:
            # Plot GEV parameter evolving over iterations
            fig,axes = plt.subplots(nrows=3, figsize=(10,18))
            handles = []
            param_names = ["Shape","Location","Scale"]
            for i_param,param in enumerate(["shape","loc","scale"]):
                ax = axes[i_param]
                for i_seed,seed in enumerate(seeds):
                    color=plt.cm.Set1(i_seed/len(seeds))
                    hteams, = xr.plot.plot(gevpar_prog[seed].sel(param=param), x="N", color=color, label=f"TEAMS {seed}", ax=ax)
                    if i_param == 0: handles.append(hteams)
                # Plot the final distribution 
                ymid = gevpar["teams"]["split"]["agg"].sel(param=param, confint=0, side=["lo"])
                ylo95,yhi95 = gevpar["teams"]["split"]["agg"].sel(param=param, confint=0.95, side=["lo","hi"]).to_numpy()
                ylo50,yhi50 = gevpar["teams"]["split"]["agg"].sel(param=param, confint=0.5, side=["lo","hi"]).to_numpy()
                ax.errorbar(Nmax, ymid, yerr=np.array([ymid-ylo95,yhi95-ymid]).reshape((2,1)),color="red",linewidth=3.0,capsize=6.0,capthick=3.0)
                ax.errorbar(Nmax, ymid, yerr=np.array([ymid-ylo50,yhi50-ymid]).reshape((2,1)),color="red",linewidth=5.0,capsize=6.0,capthick=3.0)
                print(f"dns {param} is {gevpar['dns'].sel(param=param)}")
                ymid = gevpar["dns"].sel(param=param,confint=0,side="lo")
                hdns = ax.axhline(ymid, color="black", linestyle="--", linewidth=4, label="DNS")
                if i_param == 0: handles.append(hdns)
                ylo95,yhi95 = gevpar["dns"].sel(param=param,confint=0.95,side=["lo","hi"]).to_numpy()
                ylo50,yhi50 = gevpar["dns"].sel(param=param,confint=0.5,side=["lo","hi"]).to_numpy()
                ax.fill_between(np.arange(Nmax), ylo50, yhi50, color="gray", alpha=0.25, zorder=-2)
                ax.fill_between(np.arange(Nmax), ylo95, yhi95, color="gray", alpha=0.5, zorder=-3)
                ax.set_xlabel("Number of trajectories")
                ax.set_ylabel(param_names[i_param])
                ax.set_title("")
            axes[0].legend(handles=handles,loc="lower left")
            fig.savefig(join(savefolder,"gev_param_prog"), bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)



            # Plot final GEV parameter estimates for each method
            fig,axes = plt.subplots(nrows=3,figsize=(10,18))
            gevpar_list = [gevpar["teams"]["init"]["agg"],gevpar["teams"]["split"]["agg"],gevpar["dns"]]
            tick_label = ["TEAMS Init","TEAMS","DNS"]
            for i_param,param in enumerate(["shape","loc","scale"]):
                ax = axes[i_param] # shape
                ax.bar(
                        np.arange(1,4), 
                        [rrr.sel(param=param,confint=0,side="lo") for rrr in gevpar_list], 
                        yerr=np.array([rrr.sel(param=param,confint=0.95).to_numpy().flatten() for rrr in gevpar_list]).T, 
                        tick_label=tick_label)
                ax.set_ylabel(param)
            fig.savefig(join(savefolder,"gev_paramest_comparison"), bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)

            #sys.exit()



        return

    @classmethod
    def plot_level_progress(cls, manager, savefolder, seed):
        # Level progress and diversity loss
        fig,axes = plt.subplots(nrows=2, figsize=(6,12), sharex=True)
        levels = manager.acq_state_global["levels"]
        # first panel: level progress over course of iterations
        ax = axes[0]
        ax.plot(np.arange(len(levels)), levels, marker="o", color="black")
        ax.set_xlabel("Elimination round")
        ax.set_ylabel("Level")
        
        # second panel: number of surviving families after each iteration
        ax = axes[1]
        A = manager.ens.construct_descent_matrix()
        n_anc = manager.algo_params["politics"]["base_size"]
        max_desc_scores = np.max(A[:n_anc]*manager.max_scores, axis=1)
        num_surviving_anc = np.sum((np.subtract.outer(max_desc_scores, levels) > 0), axis=0)
        ax.plot(np.arange(len(levels)), num_surviving_anc, color="black", marker="o")
        ax.set_xlabel("Elimination round")
        ax.set_ylabel("Diversity (# surviving ancestors)")

        fig.savefig(join(savefolder,f"level_progression_seed{seed}"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        return

    @classmethod
    def plot_descendant_spaghetti(cls, manager, member, savefolder, seed, abbrv, paramdisp, ylim=None):
        # Take all the descendants of a given ancestor and plot them together on the same graph, along with score as a function of iteration 
        A = manager.ens.construct_descent_matrix()
        ancestor = manager.ens.address_book[member][0]

        descendants = manager.ens.address_book[member][1:] #np.where(A[ancestor])[0]
        if len(descendants) == 0:
            raise Exception("Spaghetti only valid for multiple noodles")
        all_descendants = np.where(A[ancestor,:])[0][1:]
        descendant_total_rank = np.argmin(np.abs(np.subtract.outer(descendants,all_descendants)), axis=1)
        tu = manager.ens.model_params["time_unit"]
        delta = manager.algo_params['advance_split_time_range'][0] * tu
        tphys = manager.scores_single[ancestor]["time"].to_numpy() * tu
        tphys -= (delta + tphys[0])
        xlim = [0,tphys[-1]]
        #print(f"{xlim = }\n{tphys = }")

        fig,axes = plt.subplots(ncols=2,figsize=(12,5),sharey="row",gridspec_kw={"width_ratios": [1,1]})
        fig.set_facecolor("white")
        fig.suptitle(r"%s, Run %d, Ancestor %d"%(paramdisp.replace("\n",", "),seed,ancestor))


        # plot the scalar score
        colors = plt.cm.rainbow(np.arange(len(descendants))/(len(descendants)-1))
        ax = axes[0]
        for i_desc,desc in enumerate(descendants):
            hdesc, = xr.plot.plot(manager.scores_single[desc].assign_coords(time=tphys), x="time", linestyle="-", color=colors[i_desc], ax=ax, label="Descendants", zorder=-i_desc)
            #hgoal = ax.axhline(manager.goals[desc], color=colors[i_desc], linestyle="--")
            ax.plot(tphys[manager.max_score_tidx[desc]], manager.max_scores[desc], marker="o", markersize=12, markerfacecolor="None", markeredgecolor=colors[i_desc], markeredgewidth=3)
        hanc, = xr.plot.plot(manager.scores_single[ancestor].assign_coords(time=tphys), x="time", linestyle="--", linewidth=3, color="black", ax=ax, label=f"Ancestor {ancestor}")
        ax.plot(tphys[manager.max_score_tidx[ancestor]], manager.max_scores[ancestor], marker="o", markersize=12, markerfacecolor="None", markeredgecolor="black", markeredgewidth=3)
        ax.set_xlabel(f"Time")
        ax.set_ylabel(r"Score $R(X(t))$")
        ax.set_title(f"")
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)

        # Plot the progression of scores and the raising of the levels
        ax = axes[1]
        hscore = ax.scatter(1+np.arange(len(all_descendants)), manager.max_scores[all_descendants], c="gray", marker=".", label=r"Max score $S(X)$")
        for i_desc,desc in enumerate(descendants):
            ax.plot(1+descendant_total_rank[i_desc], manager.max_scores[desc], marker="o", markersize=12, markerfacecolor="None", markeredgecolor=colors[i_desc], markeredgewidth=3)
        ax.plot(0,manager.max_scores[ancestor], markerfacecolor="None", markeredgecolor="black", marker="o",markersize=12,markeredgewidth=3)
        # Plot goals for ALL descendants of this ancestor
        hgoal, = ax.plot(np.arange(1,len(all_descendants)+1), manager.goals[all_descendants], color="gray", linestyle="--", label="Levels")
        ax.set_xlabel("Descendant")
        ax.set_ylabel(r"Max score $S(X)$")
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        ax.yaxis.set_tick_params(which="both",labelbottom=True)
        #if ylim is not None: ax.set_ylim(ylim)
        ax.legend(handles=[hgoal],loc="lower right")
        # Make figure sub-labels
        fig.savefig(join(savefolder,f"spaghetti_{abbrv}_{len(descendants)}desc"), **svkwargs)
        plt.close(fig)



        return

    @classmethod
    def plot_family_tree_rose(cls, manager, ancestor, fig, ax):
        A = manager.ens.construct_descent_matrix()
        A1 = manager.ens.construct_descent_matrix(level=1)
        descendants = np.where(A[ancestor])[0]
        num_children_plotted = np.zeros(len(descendants), dtype=int)
        num_children_total = np.sum(A1[descendants], axis=1)

        # Decide on a position for each node 
        addrs = [manager.ens.address_book[desc] for desc in descendants]
        addr_lens = np.array([len(addr) for addr in addrs])
        unique_lengths,counts = np.unique(addr_lens,return_counts=True)

        radius = addr_lens - 1
        angle = np.zeros(len(descendants))
        room_for_children = np.zeros(len(descendants))
        room_for_children[0] = 2*np.pi

        if fig is None or ax is None:
            fig,ax = plt.subplots()
            fig.set_facecolor("white")



        for i_desc in range(1,len(descendants)):
            desc = descendants[i_desc]
            parent = addrs[i_desc][-2]
            i_parent = np.where(descendants==parent)[0][0]
            angle[i_desc] = angle[i_parent] + (num_children_plotted[i_parent]/num_children_total[i_parent]-0.5) * room_for_children[i_parent]
            num_children_plotted[i_parent] += 1
            num_children_this_generation = np.sum(addr_lens == len(addrs[i_desc]))
            room_for_children[i_desc] = np.pi/2/num_children_this_generation #min(np.pi/2,room_for_children[i_parent]/num_children_total[i_parent])
            xp = radius[i_parent] * np.cos(angle[i_parent])
            yp = radius[i_parent] * np.sin(angle[i_parent])
            xc = radius[i_desc] * np.cos(angle[i_desc])
            yc = radius[i_desc] * np.sin(angle[i_desc])
            ax.scatter([xc],[yc],color=plt.cm.jet(i_desc/len(descendants)),marker='.',s=64)
            ax.plot([xp,xc],[yp,yc],color="black")

        ax.scatter(0,0,marker="*",s=400,color="black")
        maxrad = np.max(np.abs(radius))
        ax.set_xlim([-maxrad*1.1,maxrad*1.1])
        ax.set_ylim([-maxrad*1.1,maxrad*1.1])
        for rad in np.unique(radius):
            theta = np.linspace(0,2*np.pi,100)
            ax.plot(rad*np.cos(theta),rad*np.sin(theta),linestyle="-",color="gray",zorder=-1,alpha=0.25)

        ax.xaxis.set_tick_params(which="both",labelbottom=False)
        ax.yaxis.set_tick_params(which="both",labelbottom=False)

        return











  
