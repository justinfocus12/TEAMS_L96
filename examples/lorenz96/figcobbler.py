# Ad-hoc script to cobble together figures from existing filepaths
from os.path import join, exists
import glob
from PIL import Image
import xarray as xr
import netCDF4 as nc
import numpy as np

scratch_dir = f"/net/hstor001.ib/pog/001/ju26596/splitting_results/examples/lorenz96"
date_dir = join(scratch_dir,"2023-12-27")
sub_date_dir_dns = join(date_dir,"0")
sub_date_dir_amc = join(date_dir,"1")
sub_date_dir_prt = join(date_dir,"5")
dropstr = "drop1"
base_size = 128
seedstr = f"seeds0to55"

metadir_amc = join(sub_date_dir_amc,f"meta_{dropstr}_base{base_size}")
metadir_prt = join(sub_date_dir_prt,f"meta_enssz16")

mag_list = [3.0,1.0,0.5,0.25]
tadv_list = [.0,.2,.4,.6,.8,1.0,1.2,1.4,1.6,1.8,2.0]

tododict = dict({
    "returnplots":      1,
    "perfplots":        0,
    "spaghetti":        0,
    })

delta_opt = dict({ # mapping from pairs (a,mag) to optimal advance split times
    3.0: 0.0,
    1.0: 0.6,
    0.5: 1.0,
    0.25: 1.4,
    })

# -------------- Return period plots ----------------

if tododict["returnplots"]:
    # Plot all, noise levels at a fixed delta
    for orientation in ["rl_of_rt","rt_of_rl"]:
        for tadv in tadv_list:
            imlist = []
            total_height = 0
            ylist = [] # upper left corners
            for i_mag,mag in enumerate(mag_list):
                algo_dir = join(
                        sub_date_dir_amc,
                        (f"F6p0_K40_J0_a1p0_white_wave4mag{mag}").replace(".","p"),
                        (f"AMS-AMSevw1_base{base_size}_ancdiv2_{dropstr}_horz6p0_tavg0p0_twaitdefault_splint0_adv{tadv}to{tadv}").replace(".","p"),
                        )
                returnplots = glob.glob(join(algo_dir,r"%s_%s.png"%(orientation,seedstr)))
                print(f"{returnplots = }")
                if len(returnplots) > 0:
                    img = Image.open(returnplots[0])
                    #if i_mag > 0:
                    #    img = img.crop((0,int(0.41*img.height),img.width,img.height))
                    imlist.append(img)
                    ylist.append(total_height)
                    total_height += img.height
            print(f"{ylist = }")
            print(f"{total_height = }")
            imgcat = Image.new('RGB', (img.width, total_height+1), color="white")
            for i in range(len(imlist)):
                imgcat.paste(imlist[i], (0,ylist[i]))
            imglabel = (f"{orientation}_tadv{tadv}").replace(".","p")
            imgcat.save(join(metadir_amc,f"{imglabel}.png"))

        # Plot all noise levels at (1) delta=0, and (2) their optimal deltas
        deltas2plot = dict({mag: (0.0,delta_opt[mag]) for mag in mag_list})
        for i_delta in [0,1]:
            imlist = []
            total_height = 0
            ylist = [] # upper left corners
            for i_mag,mag in enumerate(mag_list):
                delta = deltas2plot[mag][i_delta]
                algo_dir = join(
                        sub_date_dir_amc,
                        (f"F6p0_K40_J0_a1p0_white_wave4mag{mag}").replace(".","p"),
                        (f"AMS-AMSevw1_base{base_size}_ancdiv2_{dropstr}_horz6p0_tavg0p0_twaitdefault_splint0_adv{delta}to{delta}").replace(".","p")
                        )
                returnplots = glob.glob(join(algo_dir,r"%s_%s.png"%(orientation,seedstr)))
                if len(returnplots) > 0:
                    img = Image.open(returnplots[0])
                    #if i_mag > 0:
                    #    img = img.crop((0,int(0.4*img.height),img.width,img.height))
                    imlist.append(img)
                    ylist.append(total_height)
                    total_height += img.height
            print(f"{ylist = }")
            print(f"{total_height = }")
            imgcat = Image.new('RGB', (img.width, total_height+1), color="white")
            for i in range(len(imlist)):
                imgcat.paste(imlist[i], (0,ylist[i]))
            imglabel = (f"{orientation}_delta{i_delta}").replace(".","p")
            imgcat.save(join(metadir_amc,f"{imglabel}.png"))

# -------------- performance vs parameters ---------------
if tododict["perfplots"]:
    performance_metric_groups = [
            ["err_quantl1","err_quantl2"],
            ["err_quantl1_sep","err_quantl2_sep"],
            ["err_quantl1_extrap","err_quantl2_extrap"],
            ["err_quantl1_extrap_sep","err_quantl2_extrap_sep"],
            ["meanmaxgain",],
            ["err_quantl2_extrap_sep","err_quantl2_extrap_sep_std","meanmaxgain"],
            ]
    for i_gr,gr in enumerate(performance_metric_groups):
        performance_plots = [
                join(metadir_amc, f"performance_{metric}_evw1_mag_adv.png") 
                for metric in gr
                ]
        imlist = []
        total_width = 0
        xlist = []
        for i,fname in enumerate(performance_plots):
            img = Image.open(fname)
            if i > 0:
                img = img.crop((int(0.18*img.width),0,img.width,img.height))
            xlist.append(total_width)
            total_width += img.width
            imlist.append(img)
        imgcat = Image.new('RGB', (total_width+1, img.height), color="white")
        for i,img in enumerate(imlist):
            imgcat.paste(imlist[i], (xlist[i],0))
        imglabel = f"performance_columns_group{i_gr}.png"
        imgcat.save(join(metadir_amc,f"{imglabel}.png"))


# ------------- Spaghetti plots --------------------
# Zero advance split time
# pairs are (seed number, number of descendants)
if tododict["spaghetti"]:
    spaghetti_stories = xr.DataArray(
            coords={"mag": [3.0,1.0,0.5,0.25], "delta": tadv_list, "identifier": [0,1]}, # seed, numdesc
            dims=["mag","delta","identifier"],
            data=-9999
            )
    spaghetti_stories.loc[dict(mag=3.0,delta=0.0,identifier=0)] = 14
    spaghetti_stories.loc[dict(mag=3.0,delta=0.0,identifier=1)] = 7
    spaghetti_stories.loc[dict(mag=3.0,delta=0.2,identifier=0)] = 39
    spaghetti_stories.loc[dict(mag=3.0,delta=0.2,identifier=1)] = 6

    spaghetti_stories.loc[dict(mag=1.0,delta=0.0,identifier=0)] = 17
    spaghetti_stories.loc[dict(mag=1.0,delta=0.0,identifier=1)] = 9
    spaghetti_stories.loc[dict(mag=1.0,delta=0.6,identifier=0)] = 19
    spaghetti_stories.loc[dict(mag=1.0,delta=0.6,identifier=1)] = 7

    spaghetti_stories.loc[dict(mag=0.5,delta=0.0,identifier=0)] = 19
    spaghetti_stories.loc[dict(mag=0.5,delta=0.0,identifier=1)] = 7
    spaghetti_stories.loc[dict(mag=0.5,delta=1.0,identifier=0)] = 13
    spaghetti_stories.loc[dict(mag=0.5,delta=1.0,identifier=1)] = 6

    spaghetti_stories.loc[dict(mag=0.25,delta=0.0,identifier=0)] = 11
    spaghetti_stories.loc[dict(mag=0.25,delta=0.0,identifier=1)] = 6
    spaghetti_stories.loc[dict(mag=0.25,delta=1.4,identifier=0)] = 15
    spaghetti_stories.loc[dict(mag=0.25,delta=1.4,identifier=1)] = 5


    print(f"{spaghetti_stories = }")
    deltas2plot = dict({mag: (0.0,delta_opt[mag]) for mag in mag_list})
    
    imlists = [[],[]] # delta=0, delta=delta_opt
    for mag in mag_list:
        for i_delta,delta in enumerate(deltas2plot[mag]):
            algo_dir = join(
                    sub_date_dir_amc,
                    (f"F6p0_K40_J0_a1p0_white_wave4mag{mag}").replace(".","p"),
                    (f"AMS-AMSevw1_base{base_size}_ancdiv2_{dropstr}_horz6p0_tavg0p0_twaitdefault_splint0_adv{delta}to{delta}").replace(".","p")
                    )
            print(f"{spaghetti_stories.sel(dict(mag=mag,delta=delta)) = }")
            seed = spaghetti_stories.sel(mag=mag,delta=delta,identifier=0).item()
            numdesc = spaghetti_stories.sel(mag=mag,delta=delta,identifier=1).item()
            filename = join(algo_dir,f"spaghetti_bestimp_seed{seed}_{numdesc}desc.png")
            imlists[i_delta].append(Image.open(filename))
    
    for i_delta in [0,1]:
        print(f"{len(imlists[i_delta]) = }")
        w,h = imlists[i_delta][0].width,imlists[i_delta][0].height
        imgcat = Image.new("RGB",(w+1,4*h+1), color="white")
        xs = [0,0,0,0]
        ys = [0,h,2*h,3*h]
        for i in range(4):
            imgcat.paste(imlists[i_delta][i], (xs[i], ys[i]))
        imgcat.save(join(metadir_amc,f"spaghetti_delta{i_delta}.png"))




    

        


