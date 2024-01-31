import numpy as np
from numpy.random import default_rng
import pickle
import sys
import os
from os.path import join, exists
from os import mkdir, makedirs
import shutil
import copy as copylib
from multiprocessing import pool as mppool
from abc import ABC, abstractmethod



class Ensemble(ABC):
    def __init__(self,dirs,model_params,ensemble_size_limit):

        self.ensemble_size_limit = ensemble_size_limit

        # Set up directories
        if not np.all([dirname in dirs for dirname in ["home","work","output"]]):
            raise Exception(f"The list of directories is incomplete: you passed dirs = {dirs}")
        for directory in ["work","output"]:
            if exists(dirs[directory]):
                raise Exception(f"The intended {directory} directory {dirs[directory]} already exists.")
            os.makedirs(dirs[directory], exist_ok=False)
        self.dirs = dirs.copy()

        # Initialize lists to keep track of members
        self.mem_list = []
        self.address_book = []
        self.num_children = np.zeros(0, dtype=int)

        # Initialize physical parameters and underlying code infrastructure
        self.model_params = model_params.copy() # May include source files to run code, namelists, input files, diagnostic tables, number of processors
        self.setup_model() 
        
        return

    @abstractmethod
    def setup_model(self):
        # Compile source code, populate work and output directories, move executables over, whatever
        pass

    def initialize_new_member(self, EnsMemSubclass, warmstart_info, i_parent=None):
        i_mem = len(self.mem_list)
        mem_dirs = dict({
            "output": join(self.dirs["output"], f"mem{i_mem:03}"),
            "work": join(self.dirs["work"], f"mem{i_mem:03}"),
            })
        # TODO: put the seed in warmstart_info, to allow the algorithm manager to control randomness more specifically.
        new_mem = EnsMemSubclass(reconstitute=False, dirs=mem_dirs, model_params=self.model_params, warmstart_info=warmstart_info) # Warmstart info may contain the ancestral initialization file the time origin, and whether and when to perturb.
        mem_address = [i_mem]
        if i_parent is not None:
            mem_address = self.address_book[i_parent] + mem_address
            self.num_children[i_parent] += 1

        # increment the member info
        self.mem_list += [new_mem]
        self.address_book += [mem_address]
        self.num_children = np.concatenate((self.num_children, [0]))
        return

    def construct_transition_matrix(self):
        # A[i,j] = 1/N if j is a child of i (and i has N total children)
        n_mem = len(self.mem_list)
        A = np.zeros((n_mem,n_mem), dtype=int)
        for i_mem in range(n_mem):
            addr = self.address_book[i_mem]
            if len(addr) > 1:
                A[addr[-2],i_mem] += 1
        # Normalize A 
        Anorm = np.diag(1.0/np.maximum(np.sum(A, axis=1), 1)) @ A
        return A,Anorm

    def construct_descent_matrix(self,level="original"):
        # All descendants from every given ancestor
        # A[i,j] = 1 if j is a descendant of i
        n_mem = len(self.mem_list)
        A = np.zeros((n_mem,n_mem), dtype=int)
        for i_mem in range(n_mem):
            addr = np.array(self.address_book[i_mem])
            if level == "original":
                A[addr[0],i_mem] = 1
            elif isinstance(level,int) and len(addr) > level:
                A[addr[-level-1],i_mem] = 1
            elif level == "all":
                A[addr,i_mem] = 1
        return A

    def run_batch(self, memidx2run, num_chunks_per_mem, verbose=True):
        # For all member indices in memidx2run, integrate them forward 
        pool_args = [(self.mem_list[memidx2run[i]],num_chunks_per_mem[i],verbose)
                for i in range(len(memidx2run))]
        #print(f"Starting the integration with processes = {min(self.ensemble_size_limit, len(memidx2run))}...", end="")
        if self.model_params["parallel_flag"]:
            with mppool.Pool(processes=min(self.ensemble_size_limit, len(memidx2run))) as pool:
                grownup_children = pool.starmap(EnsembleMember.integrate, pool_args)
        else:
            grownup_children = [EnsembleMember.integrate(*pa) for pa in pool_args]
        print(f"Finished the integration")
        os.chdir(self.dirs["home"])
        for i in range(len(memidx2run)):
            self.mem_list[memidx2run[i]] = grownup_children[i]
            grownup_children[i].cleanup_directories()
        return 

    @abstractmethod
    def load_member_ancestry(self, i_mem_leaf):
        # Assemble the member's own lifetime and concatenate to its ancestors, in some form, e.g., as a Dask array or simply as a numpy array
        pass

    def apply_path_functional(self, pathfunc):
        # func2avg should be scalar
        funclist = []
        for i_mem in range(len(self.mem_list)):
            hist_mem = self.load_member_ancestry(i_mem)
            funclist.append(pathfunc(hist_mem))
        return funclist

    
class EnsembleMember(ABC):
    @classmethod
    def integrate(cls, mem, num_time_chunks,verbose=True):
        for i_chunk in range(num_time_chunks):
            mem.run_one_cycle(verbose=verbose)
        return mem

    def __init__(self, reconstitute=False, **kwargs):
        if reconstitute:
            self.__dict__.update(kwargs)
        else:
            # Set up directories 
            for dirtype in ["work","output"]:
                os.makedirs(kwargs["dirs"][dirtype], exist_ok=(dirtype != "work"))
            self.dirs = kwargs["dirs"].copy()
            
            # Create the directory structure
            self.setup_directories()

            # Set up model-specific information
            # model_params has the physics
            # warmstart_info has the random seeds etc. specific to a run
            self.set_run_params(kwargs["model_params"],kwargs["warmstart_info"])

        return
    @abstractmethod
    def set_run_params(self,model_params,warmstart_info):
        # warmstart_info says which file to restart from (if any), and whether to perturb the initial conditions after each restart.
        pass
    @abstractmethod
    def cleanup_directories(self):
        pass
    @abstractmethod
    def setup_directories(self):
        pass
    @abstractmethod
    def run_one_cycle(self):
        # Integrate for a single restart, and write out the corresponding history files
        # This is where we make use of a "model", if it exists
        pass
    @abstractmethod
    def load_history_selfmade(self):
        # Assemble the member's own lifetime, in some form, e.g., as a Dask array or simply as a numpy array
        pass
    @classmethod
    def distance_metric(cls, x0, x1):
        # Measure some form of distance between two different states at a fixed time 
        pass

