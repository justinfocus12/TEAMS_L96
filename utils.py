import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.special import softmax, logsumexp
from scipy.special import gamma as GammaFunction
from scipy.stats import beta as spbeta, norm as spnorm, genextreme as spgex, expon as spexp, uniform as spunif, binom as spbinom
import scipy.sparse as sps
from scipy.optimize import fsolve,bisect
from numpy.random import default_rng
import xarray as xr
import sys
import os
from os.path import join, exists
from PIL import Image

def dot2p(decnumber):
    if isinstance(decnumber,int):
        numstr = f"{decnumber}"
    else:
        numstr = f"{decnumber:.2f}".replace(".","p")
    return numstr


def my_rolling_average(da, window_size):
    dt = da["time"][1].item() - da["time"][0].item()
    nshift = int(round(window_size/dt))
    print(f"nshift = {nshift}; i_shift = ", end="")
    da_rollavg = da.copy()
    for i_shift in range(1,nshift):
        da_rollavg += da.shift(time=1)
        print(f"{i_shift}, ", end="")
    da_rollavg *= 1.0/nshift
    return da_rollavg

def rolling_average(da, window_size):
    dt = da["time"][1].item() - da["time"][0].item()
    nshift = int(round(window_size/dt))
    da_rollavg = da.rolling({"time": nshift}).mean()
    return da_rollavg

# function for determining speedup
def log_speedup(N,K,J):
    # N: number of initial ensemble members
    # K: number to drop at each generation
    # J: number of iterations
    logs = -np.log(1 + J*K/N) - J*np.log(1 - K/N)
    return logs

def iters_from_speedup(N,K,s,reps=1):
    # Given a desired speedup, solve for the number of 
    # iterations numerically with bisection
    logs_target = np.log(s) + np.log(reps)
    Jlo = 1
    Jhi = 2
    logs = log_speedup(N,K,Jhi)
    while logs < logs_target:
        Jhi *= 2
        logs = log_speedup(N,K,Jhi)
    while Jhi - Jlo > 1:
        Jmid = int(round((Jlo + Jhi)/2))
        logs_mid = log_speedup(N,K,Jmid) 
        if logs_mid < logs_target:
            Jlo = Jmid
        else:
            Jhi = Jmid
    return Jlo,Jhi

def argsort_with_random_tiebreaking(x, rng=None):
    # x is a numpy array
    # sort it, but randomly permute the repeated elements
    if rng is None:
        rng = default_rng(seed=8675309)
    xu,counts = np.unique(x, return_counts=True)
    # xu is already sorted
    order = np.argsort(x)
    counts_cumulative = 0
    for i in range(len(xu)):
        order[counts_cumulative:counts_cumulative+counts[i]] = rng.permutation(order[counts_cumulative:counts_cumulative+counts[i]])
        counts_cumulative += counts[i]
    return order
        


    
def rolling_reduction(da, window_size, reduction, nanstart=True):
    dt = da["time"][1].item() - da["time"][0].item()
    if reduction == "time_since":
        da_reduction = (0*da).where(da)
        for i_time in range(1,da.time.size):
            if da.isel(time=i_time).item() or i_time==0:
                da_reduction[dict(time=i_time)] = 0.0
            else:
                da_reduction[dict(time=i_time)] = da_reduction.isel(time=i_time-1) + 1
        da_reduction *= dt
    else:
        nshift = min(da.time.size, int(round(window_size/dt)))
        #print(f"da timeseries length = {da.time.size}, nshift = {nshift}")
        min_periods = nshift if nanstart else 1
        if reduction == "mean":
            da_reduction = 1.0/window_size * da.rolling({"time": nshift}, min_periods=min_periods).sum() * dt # Those initial points have to be VERY lucky
        elif reduction == "min":
            da_reduction = da.rolling({"time": nshift}, min_periods=min_periods).min() # Those initial points have to be VERY lucky
    return da_reduction

def compute_block_maxima(da, block_size, thresh_A):
    candidate_starts = np.where(da.to_numpy() < thresh_A)[0]
    time = da["time"].to_numpy()
    dt = time[1] - time[0]
    window_size = int(round(block_size/dt))
    block_starts = [candidate_starts[0]]
    for i in range(1,len(candidate_starts)):
        if (candidate_starts[i] > block_starts[-1] + window_size) and (candidate_starts[i] + window_size < len(time)-1):
            block_starts.append(candidate_starts[i])
    num_blocks = len(block_starts)
    block_maxima = np.zeros(num_blocks)
    for i_block in range(num_blocks):
        block_maxima[i_block] = da.isel(
            time=slice(block_starts[i_block], block_starts[i_block]+window_size)
        ).max(dim="time").item()
    return block_starts, block_maxima

def estimate_expected_hitting_time(x, thresh_list):
    Nt = len(x)
    tau = np.nan*np.ones((Nt, len(thresh_list)))
    tau[-1,:] = np.where(x[-1] > thresh_list, 0, np.nan)
    for i in np.arange(Nt-2, -1, -1):
        tau[i,:] = np.where(x[i] > thresh_list, 0, tau[i+1,:] + 1)
    return np.nanmean(tau, axis=0)

def compute_block_maxima_nothresh(da, block_size, twait):
    time = da["time"].to_numpy()
    dt = time[1] - time[0]
    Nt = len(time)
    window_size = int(round((block_size-twait)/dt))
    num_blocks = int(Nt/window_size)
    block_starts = np.arange(num_blocks) * window_size
    block_maxima = np.nanmax(da.to_numpy()[:window_size*num_blocks].reshape((num_blocks, window_size)), axis=1) #[:,wait_window_size+1:], axis=1)
    return block_starts,block_maxima

def estimate_gev_params_progressively(Xall,logWlist,max_num_uniform=1e5,min_level=None,method="PWM"):
    # Make a continuously updating progression of GEV parameter estimates
    
    gev_params_list = []
    Nlist = [len(W) for W in logWlist]
    for i_N,N in enumerate(Nlist):
        gev_params_list.append(estimate_gev_params_one_ensemble(Xall[:N],logWlist[i_N],min_level=min_level,method=method))
    gev_params = xr.DataArray(
            coords={"N": Nlist, "param": ["shape","loc","scale"]},
            dims=["N","param"],
            data=np.array(gev_params_list))

    return gev_params

# The following two functions help solve for the shape parameter k precisely in Hosking 1985 eq 13
def pwm_shape_func(shape,b_ratio): # The function to solve: (3**shape-1)/(2**shape-1) - (3*b2-b0)/(2*b1-b0)
    if np.abs(shape) < 1e-6:
        return np.log(3)/np.log(2)*(1 + np.log(3/2)/2*shape - np.log(6)/4*shape**2) - b_ratio
        # Use local quadratic approximation
    return (3**shape - 1)/(2**shape - 1) - b_ratio

def hosking_shape_fprime_log(k, log_b_ratio):
    return np.log(3)/(3**k-1) - np.log(2)/(2**k-1)

def estimate_gev_params_one_ensemble(Xall,logWall,max_num_uniform=1e5,min_level=None,method="PWM"):

    if min_level is None: min_level = -np.inf
    idx = np.where(Xall > min_level)[0]
    X = Xall[idx]
    log_weights = logWall[idx]
    logwnorm = log_weights - logsumexp(log_weights)

    if method == "MLE":
        # Replicate samples by inflating weights to all be approximate integers
        num_uniform = int(min(1/np.exp(np.min(logwnorm)), max_num_uniform)+0.5)
        print(f"num_uniform = {num_uniform}")

        weights_inflated = np.maximum(1, (wnorm*num_uniform)).astype(int)
        X_inflated = np.repeat(X, weights_inflated)
        print(f"len(bmi) = {len(X_inflated)}")
        shape,loc,scale = spgex.fit(X_inflated)
        shape *= -1 # switch conventions 
        print(f"from {len(X_inflated)} in range ({min(X_inflated),max(X_inflated)}): shape,loc,scale = {shape,loc,scale}")
    elif method == "PWM":
        # Use the method of Hosking et al 1985
        # Estimate the first three PWMs (beta0, beta1, beta2) by (b0, b1, b2)
        order = np.argsort(X)
        logWord = logwnorm[order]
        Xord = X[order]
        logFord = np.logaddexp.accumulate(logWord) # - np.exp(logWord/2 # TODO is this the proper estimator?
        b0 = np.exp(logsumexp(logWord, b=Xord)) #np.sum(Word * Xord)
        b1 = np.exp(logsumexp(logWord + logFord, b=Xord))#np.sum(Word * Xord * Ford)
        b2 = np.exp(logsumexp(logWord + 2*logFord, b=Xord))#np.sum(Word * Xord * Ford**2)
        # Solve for the shape, location, and scale parameters. Don't use the linear approximation, but
        b_ratio = (3*b2 - b0)/(2*b1 - b0)
        if b_ratio <= 0.0:
            # TODO come up with the best possible alternative...xi is a very large number, probably 
            raise Exception(f"The L-moment method has no solution; {b_ratio = }")
        # Choose initialization for solver
        tol = 1e-2
        psf0 = pwm_shape_func(0,b_ratio) 
        if psf0 == 0:
            shape = 0.0
        elif psf0 < 0: # shape > 0
            lower = 0.0
            upper = 1.0
            while pwm_shape_func(upper,b_ratio) < 0.0:
                upper *= 2.0
        else: # shape < 0
            lower = -1.0
            upper = 0.0
            while pwm_shape_func(lower,b_ratio) > 0.0:
                lower *= 2.0
        shape,root_result = bisect(pwm_shape_func, lower, upper, args=(b_ratio,), full_output=True, disp=True)

        g = GammaFunction(1 - shape)
        if shape == 0:
            scale = (2*b1 - b0)/np.log(2)
            loc = b0 - 0.5772*scale
        else:
            scale = shape*(2*b1 - b0)/((2**shape-1) * g)
            loc = b0 + scale*(1 - g)/shape


    gev_params = np.array([shape,loc,scale])
    #print(f"After fitting with method {method}, (shape,loc,scale) = \n{gev_params}")
        
    return gev_params

def gev_cdf(xin, shape, loc, scale):

    issc = np.isscalar(xin)
    x = np.array([xin]) if issc else xin

    if shape == 0:
        F = np.exp(-np.exp(-(x-loc)/scale))
    else:
        F = np.zeros_like(x)
        arg = 1 + shape*(x - loc)/scale
        idx_pos = np.where(arg > 0)[0]
        if len(idx_pos) > 0:
            F[idx_pos] = np.exp(-arg[idx_pos]**(-1/shape))
        if len(idx_pos) < len(x):
            F[np.setdiff1d(range(len(x)),idx_pos)] = 1.0 if shape < 0 else 0.0
    if issc: F = F[0]
    return F

def gev_lsf(xin, shape, loc, scale):
    issc = np.isscalar(xin)
    x = np.array([xin]) if issc else xin

    if shape == 0:
        F = np.log(-np.expm1(-np.exp(-(x-loc)/scale)))
    else:
        F = np.zeros_like(x)
        arg = 1 + shape*(x - loc)/scale
        idx_pos = np.where(arg > 0)[0]
        if len(idx_pos) > 0:
            F[idx_pos] = np.log(-np.expm1(-arg[idx_pos]**(-1/shape)))
        if len(idx_pos) < len(x):
            F[np.setdiff1d(range(len(x)),idx_pos)] = 1.0 if shape < 0 else 0.0
    if issc: F = F[0]
    return F
    

def gev_invcdf(u, shape, loc, scale):
    if shape == 0:
        x = loc - scale * np.log(-np.log(u))
    else:
        x = loc + scale/shape * (np.power(-np.log(u), -shape) - 1)
    return x

def gev_invlsf(u, shape, loc, scale):
    # = gev_invcdf(1 - exp(u))
    return gev_invcdf(-np.expm1(u), shape, loc, scale)

def pwm_unit_test(shape, loc, scale, seed=48732, nsamp_max=1000, ntrials_per_samp=10):
    # Generate RVs from distributions with known GEV parameters (say, the three parent families themselves) and then test the fitting routine 

    rng = default_rng(seed)

    # True parameters
    param_symbols = [r"$\xi$",r"$\sigma$",r"$\mu$"]
    param_names = ["Shape", "Location", "Scale"]
    gevpar = dict(shape=shape, loc=loc, scale=scale)
    
    nsamp_list = np.exp(np.linspace(2, np.log(nsamp_max), 5)).astype(int)
    gevpar_hat = xr.DataArray(
            coords={"N": nsamp_list, "trial": np.arange(ntrials_per_samp), "param": ["shape","loc","scale"]},
            dims=["N","trial","param"],
            data=np.nan
            )
    for i_nsamp,nsamp in enumerate(nsamp_list):
        print(f"Starting sample size {nsamp}")
        for trial in range(ntrials_per_samp):
            U = rng.uniform(size=nsamp)
            X = gev_invcdf(U, shape, loc, scale)
            W = np.ones(nsamp)
            gevpar_hat.loc[dict(N=nsamp,trial=trial)] = estimate_gev_params_one_ensemble(X, W)

    fig,axes = plt.subplots(nrows=3, figsize=(10,15), sharex=True, gridspec_kw={"hspace": 0.25})
    for i_param,param in enumerate(["shape","loc","scale"]):
        ax = axes[i_param]
        htrue = ax.axhline(gevpar[param], color="black", linestyle="--", linewidth=3, label="Truth")
        for N in nsamp_list: 
            ax.scatter(N*np.ones(ntrials_per_samp), gevpar_hat.sel(N=N,param=param), color="red", marker=".")
        hpwm, = xr.plot.plot(gevpar_hat.sel(param=param).mean(dim="trial"), x="N", color="red", label="PWM", ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel("Sample size")
        ax.xaxis.set_tick_params(which="both",labelbottom=True)
        ax.set_ylabel(param_symbols[i_param])
        ax.set_title(param_names[i_param])
    fig.savefig(join("unit_tests",f"gev_fit_shp{gevpar['shape']}_loc{gevpar['shape']}_scale{gevpar['scale']}").replace(".","p"),bbox_inches="tight",pad_inches=0.2)
    plt.close(fig)
    return

def block_maxima_unit_test(seed=48732, block_size_max=10000, num_blocks_max=5000, num_trials=10):
    # Generate lots of blocks of IID random variables (TODO: also correlated random variables) to ensure a good handle on the theory 
    

    # 1. Exponential distribution should lead to Gumbel EVL
    rng = default_rng(seed=seed)
    num_blocks_list = np.exp(np.linspace(3,np.log(num_blocks_max),5)).astype(int)
    block_size_list = np.exp(np.linspace(3,np.log(block_size_max),5)).astype(int)

    U = rng.uniform(size=(num_blocks_max,block_size_max,num_trials))

    # Define inverse CDF functions for the
    rate = 25.0
    def icdf_exp(u): # F(x) = 1 - exp(-x/scale)
        return -np.log(1 - u) / rate
    beta = 3.2
    def icdf_beta(u):
        return 1 - np.power(1-u, 1/(beta-1))
    pwr = 1.1
    def icdf_powerlaw(u): # F(x) = 1 - 1/(1 + x**pwr)
        return np.power(u/(1-u), 1/pwr)

    families = ["Exponential","Beta","Powerlaw"]
    icdfs = [icdf_exp,icdf_beta,icdf_powerlaw]
    shapes = [0.0, -1.0/(beta-1), 1/pwr]
    loc_funs = [
            lambda N: np.log(N)/rate, 
            lambda N: 1 - 1/np.power(N, 1/(beta-1)),
            lambda N: np.power(N, 1/pwr),
            ]
    scale_funs = [
            lambda N: 1/rate,
            lambda N: 1/(beta-1)/np.power(N, 1/(beta-1)),
            lambda N: np.power(N, 1/pwr)/pwr,
            ]

    for i_fam in range(3):
        fam_name = families[i_fam]
        shape = shapes[i_fam]
        loc_fun = loc_funs[i_fam]
        scale_fun = scale_funs[i_fam]
        gevpar = xr.DataArray(
                coords={"block_size": block_size_list, "num_blocks": num_blocks_list, "trial": np.arange(num_trials), "param": ["shape","loc","scale"], "est": ["pwm","the"]},
                dims=["block_size","num_blocks","trial","param","est"],
                data=np.nan
                )
        print(f"About to invert the uniform to get {fam_name}...", end="")
        X = icdfs[i_fam](U)
        print(f"done")
        W = np.ones((num_blocks_max,num_trials))
        print(f"Starting loop")
        for block_size in block_size_list:
            for num_blocks in num_blocks_list:
                for trial in range(num_trials):
                    Msubset = X[:block_size,:num_blocks,trial].max(axis=0)
                    Wsubset = W[:num_blocks,trial]
                    gevpar.loc[dict(block_size=block_size,num_blocks=num_blocks,trial=trial,est="pwm")] = estimate_gev_params_one_ensemble(Msubset, Wsubset)
                    # Theoretical
                    gevpar.loc[dict(block_size=block_size,num_blocks=num_blocks,trial=trial,est="the")] = [shape, loc_fun(block_size), scale_fun(block_size)]

        # Plot estimate and ground truth
        fig,axes = plt.subplots(nrows=3, figsize=(10,15), sharex=True)
        for i_param,param in enumerate(["shape","loc","scale"]):
            ax = axes[i_param]
            handles = []
            for i_block_size,block_size in enumerate(block_size_list):
                color = plt.cm.Set1(i_block_size)
                hthe = xr.plot.plot(gevpar.sel(param=param,est="the",trial=0,block_size=block_size), x="num_blocks", color=color, linestyle="--", linewidth=3, ax=ax)
                hpwm, = xr.plot.plot(gevpar.sel(param=param,est="pwm",block_size=block_size).mean(dim="trial"), x="num_blocks", color=color, label=f"Block size {block_size}", ax=ax)
                handles.append(hpwm)
            if i_param == 0: ax.legend(handles=handles)
            ax.set_xscale("log")
            ax.set_xlabel("")
            ax.set_title("")
            ax.xaxis.set_tick_params(which="both",labelbottom=True)
            ax.set_ylabel(param)

        axes[-1].set_xlabel("Number of blocks")
        fig.savefig(join("unit_tests", f"pwm_fit_{fam_name}"), bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


    # 2. Gaussian distribution should also lead to Gumbel EVL 

    # 3. Power law distribution should lead to Frechet EVL

    # 4. Uniform distribution should lead to Weibull EVL

    return


def estimate_return_statistics_one_ensemble(func_vals_all, log_weights_all, lsf_interp=None, lev_interp=None, min_level=None):
    good_idx = np.where(np.isfinite(func_vals_all))[0]
    func_vals = func_vals_all[good_idx]
    log_weights = log_weights_all[good_idx]
    order = np.argsort(func_vals)
    lsf = np.logaddexp.accumulate(log_weights[order[::-1]])[::-1]
    lsf -= lsf[0]
    #print(f"{lsf = }")
    f = func_vals[order]
    if lsf_interp is None:
        lsf_interp = lsf
    f_interp = np.interp(lsf_interp[::-1], lsf[::-1], f[::-1])[::-1]
    # Out-of-bounds checking
    idx_roob = np.sort(np.where(lsf_interp < np.nanmin(lsf))[0])
    if len(idx_roob) > 1:
        #print(f"{idx_roob = }")
        f_interp[idx_roob[1:]] = np.nan
    idx_loob = np.sort(np.where(lsf_interp > np.nanmax(lsf))[0])
    if len(idx_loob) > 1:
        #print(f"{idx_loob = }")
        f_interp[idx_loob[:-1]] = np.nan


    if (
            (np.nanmin(f_interp) > np.nanmin(f)+1e-6 and len(idx_loob) > 0) or 
            (np.nanmax(f_interp) < np.nanmax(f)-1e-6 and len(idx_roob) > 0)
            ):
        print(f"{lsf_interp = }")
        print(f"{lsf = }")
        print(f"{f = }")
        print(f"{f_interp = }")
        raise Exception(f"f has range ({np.nanmin(f)},{np.nanmax(f)}), but f_interp has smaller range ({np.nanmin(f_interp)},{np.nanmax(f_interp)})")
    rlev = xr.DataArray(
            coords={"lsf": lsf_interp, "est": ["empirical","gev"]}, dims=["lsf","est"],
            data=np.nan)
    rlev.loc[dict(est="empirical")] = f_interp
    gevpar = estimate_gev_params_one_ensemble(func_vals,log_weights,min_level=min_level,method="PWM")
    # Infer the level corresponding to the desired cdf
    # TODO finish converting to lsf coordinates
    rlev.loc[dict(est="gev")] = gev_invlsf(lsf_interp, gevpar[0], gevpar[1], gevpar[2])

    # Now do the reverse: for a range of return levels, extract a return period
    if lev_interp is None:
        lev_interp = np.linspace(np.nanmin(f), np.nanmax(f), 30)
    rlsf = xr.DataArray(
            coords={"lev": lev_interp, "est": ["empirical","gev"]},
            dims=["lev","est"],
            data=np.nan
            )
    rlsf.loc[dict(est="empirical")] = np.interp(lev_interp, f, lsf, left=np.nan, right=np.nan)
    #print(f'{rlsf.loc[dict(est="empirical")] = }')
    rlsf.loc[dict(est="empirical")] = rlsf.sel(est="empirical").where(rlsf["lev"] <= np.max(f), other=np.nan)
    rlsf.loc[dict(est="gev")] = gev_lsf(lev_interp, gevpar[0], gevpar[1], gevpar[2])

    # TODO put confidence intervals on this 

    return rlev,rlsf,gevpar

def estimate_return_statistics_many_ensembles(func_vals_dict, log_weights_dict, conf_ranges=None, min_level=None, lsf_interp=None, lev_interp=None):
    # Assemble multiple independent weighted ensembles together to get aggregate estimates (and potentially error bars)
    seeds = list(func_vals_dict.keys())
    # Compute a point estimate from the full pooled "super-ensemble"
    logw_all = np.concatenate(tuple([log_weights_dict[seed] for seed in seeds]))
    # Set the number of equivalent IID samples...how?
    # 0. number of base members (wrong)
    #alpha_plus_beta = np.exp(logsumexp(logw_all)) # = alpha + beta
    # 1. Effective sample size
    alpha_plus_beta = np.exp(2*logsumexp(logw_all) - logsumexp(2*logw_all)) # effective sample size 
    # 2. Number of nonzero weights
    #alpha_plus_beta = np.sum(np.isfinite(logw_all))
    print(f"{len(logw_all) = }")
    print(f"{alpha_plus_beta = }")

    rlev = dict()
    rlsf = dict()
    gevpar = dict()
    rlev_boot = dict()
    rlsf_boot = dict()
    gevpar_boot = dict()
    
    # super-ensemble
    f_all = np.concatenate(tuple([func_vals_dict[seed] for seed in seeds]))
    print(f"\n\n------------MAXBUG---------------")
    print(f"{np.nanmin(f_all) = }, {np.nanmax(f_all) = }")
    print(f"\n\n------------MAXBUG---------------")
    bootstrap_params = dict({
        "num_subsets": 5000, 
        "segregated": True, 
        "chunk_sizes": [len(func_vals_dict[seed]) for seed in seeds], 
        "chunks_per_subset": len(seeds) # TODO vary this to investigate effect of smaller ensemble sizes
        })
    wilson_flag = False
    wilson_sample_size = None
    rlev["sup"],rlsf["sup"],gevpar["sup"],rlev_boot["sup"],rlsf_boot["sup"],gevpar_boot["sup"] = estimate_return_levels_and_errbars(f_all, logw_all, bootstrap_params, wilson_flag, wilson_sample_size, lsf_interp=lsf_interp, lev_interp=lev_interp, conf_ranges=conf_ranges)
    # aggregate statistics
    lsf_interp = rlev["sup"]["lsf"].to_numpy()
    lev_interp = rlsf["sup"]["lev"].to_numpy()


    rlev["sep"] = xr.DataArray(
            coords={"seed": seeds, "lsf": lsf_interp, "est": ["empirical","gev"]},
            dims=["seed","lsf","est"],
            data=np.nan)
    rlsf["sep"] = xr.DataArray(
            coords={"seed": seeds, "lev": lev_interp, "est": ["empirical","gev"]},
            dims=["seed","lev","est"],
            data=np.nan)
    gevpar["sep"] = xr.DataArray(
            coords={"seed": seeds, "param": ["shape","loc","scale"]},
            dims=["seed","param"],
            data=np.nan)
    print(f"\n\n------------MAXBUG---------------")
    for seed in seeds:
        rlev["sep"].loc[dict(seed=seed)],rlsf["sep"].loc[dict(seed=seed)],gevpar["sep"].loc[dict(seed=seed)] = estimate_return_statistics_one_ensemble(func_vals_dict[seed], log_weights_dict[seed], lsf_interp=lsf_interp,lev_interp=lev_interp)
        print(f"{seed = }, {np.nanmin(func_vals_dict[seed]) = }, {np.nanmax(func_vals_dict[seed]) = }, {np.nanmax(rlev['sep'].sel(seed=seed,est='empirical')) = }")
    print(f"\n\n------------MAXBUG---------------")
        #print(f'////////////////\n///////////\n\tSeed {seed}, \n{rlev["sep"].loc[dict(seed=seed)] = }')

    # ------- debugging -------------
    print(f"\n\n------------MAXBUG (begin)----------------")
    max_sep = rlev["sep"].sel(est="empirical").max().item()
    max_sup = rlev["sup"].sel(est="empirical",confint=0,side="lo").max().item()
    print(f"{max_sep = }, {max_sup = }")
    print(f"------------MAXBUG (end)----------------\n\n")


    # -------------------------------
    # compute confidence intervals 
    if conf_ranges is None:
        conf_ranges = np.array([0.5,0.95])
    conf_ranges = np.sort(np.union1d([0],conf_ranges))
    rlev["agg"] = xr.DataArray(
            coords={"lsf": lsf_interp, "est": ["empirical","gev"], "confint": conf_ranges, "side": ["lo","hi"]},
            dims=["lsf","est","confint","side"],
            data=np.nan)
    rlsf["agg"] = xr.DataArray(
            coords={"lev": lev_interp, "est": ["empirical","gev"], "confint": conf_ranges, "side": ["lo","hi"]},
            dims=["lev","est","confint","side"],
            data=np.nan)
    gevpar["agg"] = xr.DataArray(
            coords={"param": ["shape","loc","scale"], "confint": conf_ranges, "side": ["lo","hi"]},
            dims=["param","confint","side"],
            data=np.nan)
    rlev["agg"].loc[dict(confint=0,side="lo")] = rlev["sep"].mean(dim="seed") # or median? 
    rlev_std_err = rlev["sep"].std(dim="seed")/np.sqrt(len(seeds))
    rlsf["agg"].loc[dict(confint=0,side="lo")] = logsumexp(rlsf["sep"].to_numpy(), axis=rlsf["sep"].dims.index("seed")) - np.log(len(seeds))
    rlsf_std_err = rlsf["sep"].std(dim="seed")/np.sqrt(len(seeds)) # TODO modify to clip probabilities to (0,1)
    # Error bars on RLSF: use Wilson Score interval or Beta distribution or Bootstrap
    errbar_version = "bootstrap"
    alpha = np.exp(rlsf["sup"].sel(confint=0,side="lo") + np.log(alpha_plus_beta)) #(rlsf_point * (1 - rlsf_point) / rlsf_std_err**2 - 1) * rlsf_point
    beta = alpha_plus_beta - alpha #alpha * (1 - rlsf_point) / rlsf_point
    gevpar["agg"].loc[dict(confint=0,side="lo")] = gevpar["sup"].sel(confint=0,side="lo")
    gevpar_std_err = gevpar["sep"].std(dim="seed")/np.sqrt(len(seeds))
    print(f"rlsf_std_err = \n{rlsf_std_err}")
    print(f"rlev_std_err = \n{rlev_std_err}")
    for ci in conf_ranges[1:]:
        z = spnorm.ppf(0.5 + 0.5*ci)
        rlev["agg"].loc[dict(confint=ci,side="lo")] = rlev["agg"].sel(dict(confint=0,side="lo")) - z*rlev_std_err
        rlev["agg"].loc[dict(confint=ci,side="hi")] = rlev["agg"].sel(dict(confint=0,side="lo")) + z*rlev_std_err
        # Option 1: beta distribution
        if errbar_version == "beta":
            rlsf["agg"].loc[dict(confint=ci,side="lo")] = np.log(spbeta.ppf(0.5 - 0.5*ci, alpha, beta)) #rlsf["agg"].sel(dict(confint=0,side="lo")) - z*rlsf_std_err
            rlsf["agg"].loc[dict(confint=ci,side="hi")] = np.log(spbeta.ppf(0.5 + 0.5*ci, alpha, beta)) #rlsf["agg"].sel(dict(confint=0,side="lo")) + z*rlsf_std_err
        # Option 2: Wilson score interval
        elif errbar_version == "wilson":
            nsucc = np.exp(rlsf["sup"].sel(confint=0,side="lo") + np.log(alpha_plus_beta))
            #nsucc = nsucc.where(np.isfinite(nsucc), 0)
            #nfail = len(seeds) - nsucc
            nfail = alpha_plus_beta * (-np.expm1(rlsf["sup"].sel(confint=0,side="lo")))
            #nfail = nfail.where(np.isfinite(nfail), 0)
            lower,upper = wilson_score_interval(nsucc, nfail, ci)
            print(f"{lower = }")
            print(f"{upper = }")
            print(f"{nsucc = }")
            print(f"{nfail = }")
            rlsf["agg"].loc[dict(confint=ci,side="lo")] = np.log(lower)
            rlsf["agg"].loc[dict(confint=ci,side="hi")] = np.log(upper)
        gevpar["agg"].loc[dict(confint=ci,side="lo")] = gevpar["agg"].sel(dict(confint=0,side="lo")) - z*gevpar_std_err
        gevpar["agg"].loc[dict(confint=ci,side="hi")] = gevpar["agg"].sel(dict(confint=0,side="lo")) + z*gevpar_std_err

    print(f"gevpar['agg'].sel(confint=0.95) = {gevpar['agg'].sel(confint=0.95)}")
    #sys.exit()
    return rlev,rlsf,gevpar,rlev_boot,rlsf_boot,gevpar_boot

def estimate_return_levels_and_errbars(X, logW, bootstrap_params, wilson_flag, wilson_sample_size=None, lsf_interp=None, lev_interp=None, conf_ranges=None):
    # unstructure bootstrap for error bars
    N = len(X)
    if wilson_sample_size is None: # use effective sample size
        wilson_sample_size = np.exp(2*logsumexp(logW) - logsumexp(2*logW))
    # full point estimate
    rlev_point_estimate,rlsf_point_estimate,gevpar_point_estimate = estimate_return_statistics_one_ensemble(X,logW,lsf_interp=lsf_interp, lev_interp=lev_interp)
    #print(f"{rlsf_point_estimate = }")
    # Initialize result for full confidence intervals
    if conf_ranges is None:
        conf_ranges = np.array([0.5,0.95])
    conf_ranges = np.sort(np.union1d([0],conf_ranges))
    lsf_interp = rlev_point_estimate["lsf"].to_numpy()
    lev_interp = rlsf_point_estimate["lev"].to_numpy()
    rlev = xr.DataArray(
            coords={"lsf": lsf_interp, "est": ["empirical","gev"], "confint": conf_ranges, "side": ["lo","hi"]},
            dims=["lsf","est","confint","side"],
            data=np.nan)
    rlev.loc[dict(confint=0,side="lo")] = rlev_point_estimate.to_numpy()
    rlsf = xr.DataArray(
            coords={"lev": lev_interp, "est": ["empirical","gev"], "confint": conf_ranges, "side": ["lo","hi"]},
            dims=["lev","est","confint","side"],
            data=np.nan)
    rlsf.loc[dict(confint=0,side="lo")] = rlsf_point_estimate.to_numpy()
    gevpar = xr.DataArray(
            coords={"param": ["shape","loc","scale"], "confint": conf_ranges, "side": ["lo","hi"]},
            dims=["param","confint","side"],
            data=np.nan)
    gevpar.loc[dict(confint=0,side="lo")] = gevpar_point_estimate

    # Bootstrap estimates
    rlev_boot = xr.DataArray(
            coords={"lsf": lsf_interp, "est": ["empirical","gev"], "boot": np.arange(bootstrap_params["num_subsets"])},
            dims=["lsf","est","boot"],
            data=np.nan)
    rlsf_boot = xr.DataArray(
            coords={"lev": lev_interp, "est": ["empirical","gev"], "boot": np.arange(bootstrap_params["num_subsets"])},
            dims=["lev","est","boot"],
            data=np.nan)
    gevpar_boot = xr.DataArray(
            coords={"param": ["shape","loc","scale"], "boot": np.arange(bootstrap_params["num_subsets"])},
            dims=["param","boot"],
            data=np.nan)
    rng = default_rng(seed=28475)
    # NOTE boot_tag_sample_size refers to the number of unique tags, not necessarily the number of samples. 
    for i_boot in range(bootstrap_params["num_subsets"]):
        if bootstrap_params["segregated"]:
            assert np.sum(bootstrap_params["chunk_sizes"]) == N
            chunk_starts = np.concatenate(([0],np.cumsum(bootstrap_params["chunk_sizes"][:-1])))
            chunk_ends = chunk_starts + bootstrap_params["chunk_sizes"]
            num_chunks = len(bootstrap_params["chunk_sizes"])
            tag_subset = rng.choice(np.arange(num_chunks), size=bootstrap_params["chunks_per_subset"], replace=True)
            tag_subset_unique,tag_counts = np.unique(tag_subset,return_counts=True)
            subset = np.concatenate(tuple([
                list(range(chunk_starts[tag_subset_unique[i_tag]],chunk_ends[tag_subset_unique[i_tag]])) * tag_counts[i_tag]
                for i_tag in range(len(tag_subset_unique))
                ]))
        else:
            subset = rng.choice(np.arange(N), size=bootstrap_params["samples_per_subset"], replace=True)
        rlev_boot.loc[dict(boot=i_boot)],rlsf_boot.loc[dict(boot=i_boot)],gevpar_boot.loc[dict(boot=i_boot)] = estimate_return_statistics_one_ensemble(X[subset], logW[subset], lsf_interp=lsf_interp, lev_interp=lev_interp)
        if i_boot % 50 == 0:
            print(f"Finished boot {i_boot} out of {bootstrap_params['num_subsets']}")

    # Fill in the confidence intervals
    logp_full = rlsf.sel(est="empirical",confint=0,side="lo").to_numpy()
    for ci in conf_ranges[1:]:
        rlev.loc[dict(confint=ci,side="lo")] = rlev_boot.quantile(0.5-0.5*ci, dim="boot")
        rlev.loc[dict(confint=ci,side="hi")] = rlev_boot.quantile(0.5+0.5*ci, dim="boot")
        # for probabilities, we have two options
        if wilson_flag:
            # 1. Wilson score
            nsucc = np.nan_to_num(np.exp(logp_full + np.log(wilson_sample_size)), nan=0)
            nfail = wilson_sample_size - nsucc
            lower,upper = wilson_score_interval(nsucc, nfail, ci)
            rlsf.loc[dict(est="empirical",confint=ci,side="lo")] = np.log(lower)
            rlsf.loc[dict(est="empirical",confint=ci,side="hi")] = np.log(upper)
            print(f"{lower = }, {upper = }")
            print(f"{rlsf.sel(confint=ci) = }")
        else:
            # 2. Bootstrap (unsatisfying; CIs often don't include the point estimate)
            rlsf.loc[dict(confint=ci,side="lo")] = rlsf_boot.quantile(0.5-0.5*ci, dim="boot")
            rlsf.loc[dict(confint=ci,side="hi")] = rlsf_boot.quantile(0.5+0.5*ci, dim="boot")
        gevpar.loc[dict(confint=ci,side="lo")] = gevpar_boot.quantile(0.5-0.5*ci, dim="boot")
        gevpar.loc[dict(confint=ci,side="hi")] = gevpar_boot.quantile(0.5+0.5*ci, dim="boot")

    return rlev,rlsf,gevpar,rlev_boot,rlsf_boot,gevpar_boot 

def estimate_return_level_mbm(da, block_size, twait, boot_sample_size_list, conf_ranges=None, n_boot=100, min_quantile=0.9, lsf_interp=None, lev_interp=None, ):
    block_starts,block_maxima = compute_block_maxima_nothresh(da, block_size, twait)

    # ----------- TODO adjust for correlation structure within da. -----------


    # --------------------------------------------------
    print(f"block_maxima.shape = {block_maxima.shape}")
    print(f"{np.nanmin(block_maxima) = }, {np.nanmax(block_maxima) = }, {np.mean(np.isnan(block_maxima)) = }")
    rlev_list = []
    rlsf_list = []
    gevpar_list = []
    for bss in boot_sample_size_list:
        bootstrap_params = dict({
            "num_subsets": n_boot, # How many bootstrap resamplings to draw (the more the better)
            "segregated": False, # if True, bootstrap in segregated chunks
            # Relevant if segregated == True
            "chunk_sizes": None, 
            "chunks_per_subset": None,
            # Relevant if segregated == False
            "samples_per_subset": bss,
            })
        wilson_flag = False
        wilson_sample_size = bss
        rlev_bss,rlsf_bss,gevpar_bss,_,_,_ = estimate_return_levels_and_errbars(block_maxima, np.zeros(len(block_maxima)), bootstrap_params, wilson_flag, wilson_sample_size, lsf_interp=lsf_interp, lev_interp=lev_interp, conf_ranges=conf_ranges)
        rlev_list.append(rlev_bss)
        rlsf_list.append(rlsf_bss)
        gevpar_list.append(gevpar_bss)
    rlev = xr.concat(rlev_list,dim="bss").assign_coords(bss=boot_sample_size_list)
    rlsf = xr.concat(rlsf_list,dim="bss").assign_coords(bss=boot_sample_size_list)
    gevpar = xr.concat(gevpar_list,dim="bss").assign_coords(bss=boot_sample_size_list)
    return rlev,rlsf,block_maxima,gevpar

def wilson_score_interval(nsucc, nfail, confint):
    # compute the binomial confidence interval on p given nsucc successes and nfail failures
    z = spnorm.ppf(0.5 + 0.5*confint)
    z2 = z*z
    n = nsucc + nfail
    phat = nsucc/n
    center = (phat + z2/(2*n))/(1 + z2/n)
    radius = z/(1+z2/n) * np.sqrt(phat*(1-phat)/n + z2/(4*n*n))
    lower = center - radius
    idx0 = np.where(nsucc==0)[0]
    #lower[idx0] = 3/n[idx0]
    upper = center + radius
    idx0 = np.where(nfail==0)[0]
    #upper[idx0] = 1-3/n[idx0]
    return lower,upper


def pivotally_resample(weights, N_out, rng):
    N_in = len(weights)
    randperm = rng.permutation(np.arange(N_in))
    N_remaining = N_out
    weight_remaining = np.sum(weights)
    replicas = np.zeros(N_in, dtype=int)
    for i in randperm:
        if N_remaining > 0: 
            replicas[i] = int(N_remaining*weights[i]/weight_remaining + rng.uniform())
            N_remaining -= replicas[i]
            weight_remaining -= weights[i]
    return replicas

def pivotally_resample_unit_test():
    rng = default_rng(seed=2542)
    N_in = 10
    N_out = 20
    scores = np.sort(rng.normal(size=N_in))
    for k in range(5):
        print(f"---- k = {k} -----")
        weights = softmax(k * scores)
        print(f"weightsum = {np.sum(weights)}")
        replicas = pivotally_resample(weights, N_out, rng)
        print(f"scores = \n{scores}\nreplicas = \n{replicas}")
    return


def stack_images(input_paths, output_path):
    # This comes from ChatGPT
    images = [Image.open(path) for path in input_paths]

    # Get the size of the first image
    width, height = images[0].size

    # Create a new image with the same width and the sum of the heights
    combined_image = Image.new("RGB", (width, sum(image.height for image in images)))

    # Paste each image onto the combined image
    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.height

    # Save the combined image
    combined_image.save(output_path)
    return

def calculate_return_time_ou1d(alpha, sigma, thresh_list=None):
    # Calculate the exact (up to quadrature) the return period for the 1D OU process dX = -alpha*X dt + sigma dW
    epsilon = sigma**2/2
    prefactor = np.sqrt(2*np.pi)/alpha
    std_the = sigma/np.sqrt(2*alpha)
    if thresh_list is None:
        thresh_list = np.arange(0,10,1)*std_the
    resthe = xr.DataArray(coords={"level": thresh_list}, dims=["level"], data=np.nan)
    for i_thresh,thresh in enumerate(thresh_list):
        z = np.linspace(-thresh, thresh, 10000)*np.sqrt(alpha/epsilon)
        integrand = np.exp(z**2/2) * spnorm.cdf(z)**2
        dz = z[1] - z[0]
        integral = 0.5*np.sum(integrand[:-1] + integrand[1:]) * dz
        return_time = prefactor*integral
        resthe[dict(level=i_thresh)] = return_time
    return resthe

def compute_exceedance_prob_ou1d(alpha,sigma,T,a):
    q,t,x = compute_committor_ou1d(alpha,sigma,T,a)
    rho = np.exp(-alpha*x**2/sigma**2) / np.sqrt(np.pi*sigma**2/alpha)
    return np.sum(rho * q[0,:]) * (x[1]-x[0])


def compute_committor_ou1d(alpha,sigma,T,a):

    # b(x) = -alpha*x
    # a(x) = sigma
    # q_{-t} = b(x)q_x + diffq_xx

    std_the = sigma/np.sqrt(2*alpha)
    
    bounds = {"x": np.array([-3.0*std_the,a]), "t": np.array([0.0, T])}
    dx = 0.05 * std_the
    dt = dx / 5.0
    Nx = int(np.ceil((bounds["x"][1] - bounds["x"][0])/dx)) + 1
    Nt = int(np.ceil((bounds["t"][1] - bounds["t"][0])/dt)) + 1
    x = np.linspace(bounds["x"][0], bounds["x"][1], Nx)
    t = np.linspace(bounds["t"][0], bounds["t"][1], Nt) # This time will be reversed at the end; doesn't matter because autonomous
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    
    bwdness_diff = 0.5 # how backward to treat the diffusion
    bwdness_adv = 0.5 # how backward to treat the advection
    
    def advection(x):
        return -alpha * x
    def diffusion(x):
        return sigma**2/2 * np.ones_like(x)
    
    diff_mat = np.zeros((Nx,Nx))
    diff_mat[(np.arange(Nx),np.arange(Nx))] = -2.0
    diff_mat[(np.arange(Nx-1),np.arange(1,Nx))] = 1.0
    diff_mat[(np.arange(1,Nx),np.arange(Nx-1))] = 1.0
    diff_mat = np.diag(diffusion(x)) @ diff_mat
    
    adv_mat = np.zeros((Nx,Nx))
    adv_mat[(np.arange(Nx-1),np.arange(1,Nx))] = 0.5
    adv_mat[(np.arange(1,Nx),np.arange(Nx-1))] = -0.5
    adv_mat = np.diag(advection(x)) @ adv_mat
    
    Identity = np.eye(Nx)
    
    q = np.zeros((Nt,Nx))
    #q[0] = ((x - bounds["x"][0])/(bounds["x"][1] - bounds["x"][0]))**2
    q[0,-1] = 1.0
    
    lhsmat = Identity - adv_mat*bwdness_adv*dt/dx - diff_mat*bwdness_diff*dt/dx**2
    rhsmat = Identity + adv_mat*(1-bwdness_adv)*dt/dx + diff_mat*(1-bwdness_diff)*dt/dx**2
    
    # Fix the boundaries
    # Lower boundary: Neumann
    lhsmat[0,:] = 0.0
    lhsmat[0,:3] = np.array([-3,4,-1])
    rhsmat[0,:] = 0.0
    # Upper boundary: Dirichlet (just carry forward the boundary)
    lhsmat[Nx-1,:] = 0.0
    lhsmat[Nx-1,Nx-1] = 1.0
    rhsmat[Nx-1,:] = 0.0
    rhsmat[Nx-1,Nx-1] = 1.0
    
    for ti in range(1,Nt):
        q[ti] = np.linalg.solve(lhsmat, rhsmat @ q[ti-1])
        q[ti,Nx-1] = 1.0
    
    # Finally, reverse it 
    q = q[::-1]

    return q,t,x

if __name__ == "__main__":
    # Replace these paths with the paths to your input PNG files
    input_paths = ["image1.png", "image2.png", "image3.png", "image4.png"]

    # Replace this path with the desired output path for the combined image
    output_path = "combined_image.png"

    stack_images(input_paths, output_path)







    

if __name__ == "__main__":
    test_pwm =           0
    test_block_maxima =  1
    if test_pwm:
        for shape in [-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]:
            pwm_unit_test(shape=shape, loc=0.0, scale=1.0, nsamp_max=10000, ntrials_per_samp=20)
    if test_block_maxima:
        block_maxima_unit_test()
