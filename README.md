This repository contains python modules and scripts to run the experiments in the preprint ``Bringing statistics to storylines: rare event sampling for sudden, transient extreme events'' by Justin Finkel and Paul A. O'Gorman. 

Below we give a brief set of instructions for running the code to produce the results from the paper, followed by a longer (but still cursory) overview of the code organization with indications of how to extend it for other dynamical systems.

# How to run the code

All runnable scripts are in the subfolder `TEAMS_L96/examples/lorenz96`. Navigate there first. 

## Run a control simulation
Open `ensemble_lorenz96.py`. At the bottom `if __name__ == "__main__" block, there are two possible procedures to run: `main_dns()` which runs a straightforward control simulation with fixed parameters, and `dns_meta_analysis` which collects and compares output from several DNS runs with different parameters. 

Let's start with `main_dns()`, a function which begins with a `tododict` specifying a list of tasks to perform (in order). This way, you can quickly repeat later tasks (which often involve making figures) without repeating expensive earlier tasks, which have been saved to file. This is my way of notebook-ifying a plain `.py` file. 

First, modify `home_dir` to point to your base `TEAMS_L96` DIRECTORY, and modify `scratch_dir` to point to where you want results saved. Right below, you can further configure the `dns_supdir` file (I tack onto the end `date_str` and `sub_date_str` for the present date). Within `dns_supdir`, you can put outputs from multiple parameter settings, which will be automatically labeled in further subdirectories according to the function `label_from_config`. 

In the command line, if you type 

```
python ensemble_lorenz96.py 1
```

This will run a DNS simulation in chunks of length 2000 (modify via `duration_dns_phys`) and repeat 10 times (modify via `num_repetitions`) with stochastic forcing at wavenumber 4 (modify within `config_onetier.yml`) with magnitude equal to the `1`th argument of `siglist` and a value of `a` (the advection coefficient) equal to `1`th argument of `alist` (modify within `ensemble_lorenz96.py/main_dns()`). You could also specify all parameters statically within `config_onetier.yml`, including `a` and noise magnitude, but then in `main_dns()` you must set `loop_through_sigmas` and `params_from_sysargs` both to `False`. The output will go into `dns_dir`, a subdirectory of `dns_supdir` which is automatically labeled by `Lorenz96Ensemble.label_from_config`. After the DNS is run, a section of it will be plotted and the `.png` file saved to `dns_dir` (see the function `ensemble_lorenz96/visualize_long_integration`). 

By running the same script again, you will keep adding chunks of integration to the same output directory `dns_dir`, each one a different `Lorenz96EnsembleMember` object. The `m`th chunk will be stored in `<dns_dir>/DNS/output/mem<m>`, whereas the metadata for the whole `Lorenz96Ensemble` will be stored in `<dns_dir>/DNS/output/ens`, a binary file. See below for explanations of these objects. Fig. 3 of the paper used 64 total chunks for each parameter set.

Now, supposing you've done the above for a variety of parameters (either by modifying the `config_onetier.yml` file and running the code repeatedly, or by looping through parameters, or by running multiple parameters in a SLURM batch job (see the last line of `dns.sbatch`)), you can modify `if __name__ == __main__` to call `ensemble_lorenz96/dns_meta_analysis` to make plots like Figs. 2 and 3 of the paper. To do this, modify `forcing_dir_list` to be the list of output directories (`dns_dir` variables) from DNS that you want to analyze, and modify `meta_dir` to be the location for the intercomparison plots. `meta_dir` need not bear any relation to `forcing_dir_list`. 


# Run the TEAMS algorithm

Open `teams_manager_lorenz96.py`. As above, change `scratch_dir` to your preferred storage location, and `home_dir` to the top-level repo directory. To only run TEAMS, set `tododict['run_teams_flag'] = 1`. The following flags specify post-analysis to perform. If you want return plots comparing to DNS, meaning you set `tododict['summarize_tail'] = 1` and `tododict['plot_return_stats'] = 1`, you must also specify `dns_dir_validation` as the `<dns_dir>/DNS` generated above. 

Running TEAMS has a few more command line arguments. The first argument, like in DNS, specifies which combination of (`a`, `F_4`) is used in the model, through the index in `maglist` (same as `siglist` in `ensemble_lorenz96.py`, sorry for the inconsistency) and `alist`. There is a further second argument which specifies which advance split time (delta in the paper) to use, as an index in `tadvlist`. Finally, all remaining command line arguments (integers) give a list of seeds to use, one for each independent run of TEAMS. So, for example, to run TEAMS with `a=1`, `F_4=0.5`, and an advance split time of 1.0 for 56 independent seeds, type in the command line 

```
python teams_manager_lorenz96.py 3 5 {0..55}
```

The output will go into `scratch_dir` in a hierarchical file structure by parameter and seed. Plots from post-processing will go in the folder containing the folders `seed0`, `seed1`, .... 

The many parameters of TEAMS, other than the advance split time, can be further configured in `config_teams.yml`. Note that advance split time is overwritten after being read in, unless you disable `params_from_sysargs`, in which case you need to put in two dummy arguments before {0..55} in the command above. 






# Structure of directories and classes 

The top-level folder, `TEAMS_L96`, contains abstract class definitions for managing ensembles of trajectories of dynamical systems, whereas the subfolder `TEAMS_L96/examples/lorenz96` instantiates those classes for the Lorenz-96 model. In principle, a user can add further examples by modifying the Lorenz-96-specific code and putting it into another folder, e.g., `TEAMS_L96/examples/mymodel`. The general classes are as follows:

1. `ensemble.py` defines two abstract classes: 
    - `EnsembleMember` represents a single, unbroken forward simulation. It has abstract methods for running dynamics forward (which must be instantiated by each system separately; for Lorenz-96, it is a simple Euler-Maruyama timestep, but for complex models one could call Fortran code through a subprocess or `f2py`). It also contains metadata, such as directories for saving output and `warmstart_info`, which contains initial conditions and perturbations needed for the integration. There is also an abstract method for loading the history of an ensemble member into memory, which is useful when we have thousands of integrations that would be unwieldy or impossible to store at once. 
    - `Ensemble` represents a branching tree of `EnsembleMember`s evolving according to common dynamics.  It has an attribute `mem_list` which is a list of `EnsembleMember` instances, as well as `address_book` which is a list of lists of integers encoding relations between members. The element `address_book[i]` always ends with `i`. If `i` is an ancestral trajectory, `address_book[i]` has length 1. Otherwise, `address_book[i][-2]` is the parent of `i`, and so on until the leading element which is always an ancestor. `Ensemble` also has instance methods `initialize_new_member` and `run_batch` which instantiates new `EnsembleMember`s and incorporates them into the family tree. While the forward solving is achieved in `EnsembleMember.run_one_cycle`, the dynamics can be parameterized through the `model_params` argument to `Ensemble.__init__`, which is forwarded to all `EnsembleMember`s.

2. `teams_manager.py` defines an abstract class TEAMSManager which manages an instance of Ensemble to implement the TEAMS algorithm. The manager can be seen as a state machine which updates with each new trajectory, the update being performed by the method `take_next_step`. The algorithm's state is encoded through the following mutable instance variables:
    - `max_scores` (and other lists initialized in `TEAMSManager.__init__`) track the score functions, splitting times, and other information for each new member.
    - `acq_state_global` (a dictionary) holds the information for choosing members to split, members to kill, weights assigned to each member, multiplicities, and the current level.
    - `acq_state_local` (a dictionary with a different key for each member) holds the information needed to spawn a child from that member, most importantly the time at which the member's score crosses the current level.


3. `pert_manager.py` defines an abstract class `PERTManager` which manages an instance of `Ensemble` to perform the experiments to quantify divergence rates of trajectories. The structure is similar to that of `TEAMSManager` but with much simpler logic.
