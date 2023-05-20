import os
import glob 
import argparse 
import numpy as np
import polars as pl 
import matplotlib.pyplot as plt 
from scipy import stats 
from tqdm import tqdm 

def get_histograms(
    basename: str, 
    npydir: str,
    axs: list,
    ndist: int
) -> list:
    path = os.path.join(npydir, '*.npy')
    objs = glob.glob(path, recursive=True)

    bins = 2000
    for i,obj in enumerate(objs): 
        data = np.load(obj) 
        vartitle = os.path.basename(obj).split('.')[0]
        axs[i, ndist].hist(data, bins=bins, density=True, alpha=0.6)
        axs[i, ndist].set_title(vartitle)
        axs[i, ndist].set_xticks([])
        axs[i, ndist].set_yticks([])
        axs[i, ndist].axvline(x=0, c="red")
        if i == 0: 
            axs[i, ndist].set_title(basename + "\n" + vartitle )

def main(config:dict) -> None: 
    path = os.path.join(config.input_dir, '*.ply')
    objs = glob.glob(path, recursive=True)
    
    ndist = len(objs) 

    basename = os.path.basename(objs[0]).split('.')[0]
    npdata_dir = os.path.join(config.npy_dir,basename)
    path = os.path.join(npdata_dir, '*.npy')
    vectors = glob.glob(path, recursive=True)

    nhist = len(vectors) 

    fig, axs = plt.subplots(ncols=ndist,nrows=nhist,
                            figsize=(32,32))

    until = None
    if __debug__: 
        until = 1
    distortions = [[] for _ in range(len(objs[:until]))]
    for i, obj in enumerate(tqdm(objs[:until])):
        basename = os.path.basename(obj).split('.')[0]
        npdata_dir = os.path.join(config.npy_dir,basename)
        get_histograms(basename, npdata_dir,axs,i) 

    fig.tight_layout(pad=5.0)
    fig.savefig('all_new_metrics.png', dpi=300, bbox_inches='tight')

    plt.show()
            

            

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='./models'
                        )
    parser.add_argument('--npy-dir', type=str, 
                        default='./rawnp'
                        )
    config = parser.parse_args()

    main(config) 
