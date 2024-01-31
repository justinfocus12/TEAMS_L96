This repo has many vestigial (or at least hibernating) files. The general structure is as follows.

The top-level directory ("splitting") has three key files:
1. ensemble.py contains class definitions for the Ensemble and EnsembleMember classes. These are abstract classes for storing collections of dynamical trajectories, including relations between ancestors and descendants. Each instantiation is responsible for implementing forward simulation and storage/accessing of model output. 
2. utils.py has a collection of useful functions used throughout the repo.
3. amc_manager.py contains a class definition for AMCManager, which carries out an Adaptive Monte Carlo algorithm by directing an Ensemble object. AMC includes AMS as a special case, as well as (in development) a Multi-Armed Bandit optimization algorithm. There are many other <...>_manager.py files corresponding to past experiments.

Within the "examples" directory are a collection of test cases. The three most important so far are 
1. lorenz96
2. ornstein_uhlenbeck 
3. frierson_gcm 

but (3) is not up to date.

---------------------
Software requirements
---------------------
Python >= 3.8, along with many python libraries, notably xarray, available through anaconda. I think the following packages should be sufficient for the basic test cases of OU and Lorenz96:

numpy
xarray
dask
netCDF4 
scipy
matplotlib
sklearn




----------------------
OU Process directions
----------------------
1. Navigate to examples/ornstein_uhlenbeck
2. Open the file ornstein_uhlenbeck.py, which implements the OrnsteinUhlenbeckEnsemble(Member) classes. Go to the function "run_long_integration" and change home_dir and scratch_dir to your own source directory (top level of "splitting") and where you want to store the output data, respectively. 
3. From the terminal, type 

$ python ornstein_uhlenbeck.py 

to run an integration for 30000 time units with a save-out time of 0.01 (and a timestep of 0.001). These parameters can be changed. The output folder you specified should now contain a folder called ctrl_1x3000000 (the factor of 100 counts all the timesteps), and subdirectories "output" and "work". Within "output" you should see a binary file, "ens", with the metadata for the 1-member ensemble you created by this simulation. The subdirectory "mem000" should have a netcdf file for the time history, and a "restart" file, mimicking the GCM structure. 

4. Copy the full file path to the netcdf file. In the source file amc_manager_ou.py in the function "amc" which (under its current settings) implements AMS, assign the variable "dns_file" to that path. Modify home_dir and scratch_dir as above. Note that the amc() routine automatically appends some extra stuff to scratch_dir to specify this unique experiment, including "date_str",  "sub_date_str", and "AMS_<jumbled abbreviations for parameters>". The variable 'expt_dir' specifies the path to that directory. 

5. Adjust parameters of the algorithm via the dictionary "algo_params" to your liking. Most important parameters are "ams_base_size" (initial ensemble size) and "ams_num2drop" (how many members to kill each generation)

6. In the "if __name__ == "__main__" block, set run_amc_flag = 1 and analyze_amc_flag = 0. The latter is for making plots after the run is over.

7. In the terminal, type

$ python amc_manager_ou.py 0 1

to run two independent runs of AMS with random seeds 0 and 1. You can add more numbers to the list if you want. Lots and lots of output will be printed to the screen. Afterward, the metadata and trajectory data will be stored in expt_dir.

8. To analyze the output, copy the full path to expt_dir ("<date_str>/<sub_date_str>/AMS_<...>") and assign it to the variable "expt_dir" within the "amc_analysis" function. Assign "ctrl_dir" to the path to the DNS you generated (stopping before the "output" directory). In the "if __name__ == "__main__" block, set run_amc_flag = 0 and analyze_amc_flag = 1. 

9. In the terminal, type 

$ python amc_manager_ou.py 

and see the resulting summary plots of the algorithm in expt_dir. 


-----------------------
Lorenz-96 directions
-----------------------



