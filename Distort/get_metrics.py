#region imports 
from pip._vendor.rich.progress import Progress
import argparse 
import open3d as o3d
import numpy as np
import multiprocessing as mp 
import functools 
import glob 
import subprocess 
import polars as pl 
import os 
import utils.metrics as pcmm
from tqdm import tqdm 
#endregion imports 

METRICS = { 
    'octree': [0, 6, "run_pcc/h. (p2plane)"],
    'downsample': [7, 13, "run_pcc/mseF (p2point)"], 
    'localoffset': [14, 20, "run_pcc/mseF (p2point)"],
    'localrotation': [21, 27, "run_pcc/mseF (p2point)"],
    'gaussianshift': [28, 34, "run_pcc/h. (p2plane)"],
}
#endregion GLOBALS 

def get_correct_metric(config): 
    global METRICS
    ref_path = os.path.join(config.ref_dir, '*.ply')
    ref_objs = glob.glob(ref_path, recursive=True)

    references = []
    distortions = []
    for obj in ref_objs: 
        ref_base = os.path.basename(obj)
        ref_name = os.path.basename(obj).split('.')[0]
        dis_path = os.path.join(config.dis_dir, ref_name, '**/*.ply')
        dis_obj = glob.glob(dis_path, recursive=True)

        references += [ref_base for dis in dis_obj] 
        distortions += [os.path.basename(dis) for dis in dis_obj] 

    df = pl.DataFrame(zip(references, distortions), schema={'reference': pl.Utf8, 'distortion': pl.Utf8}) 
    df = df.with_columns(pl.lit(0).alias('metric')) 
    df = df.with_columns(pl.col('distortion').str.split('_').arr.get(1).str.split('.').arr.get(0).cast(pl.Int64).alias('order'))
    df = df.sort(['reference', 'order', 'distortion'])
    for key in list(METRICS.keys()): 
        # Get the lower bound, upper bound and function from the metrics 
        values = METRICS[key]
        # Get the function that estimates the quality error[last pos]
        fvalues = values[2].split("/") 
        fname = fvalues[0]
        args = None 
        if len(fvalues) > 1: 
            args = fvalues[1]
        func = getattr(pcmm, fname) 
        
        vrange = [_ for _ in range(values[0], values[1]+1)]
        print(vrange) 
        df = ( 
            df
            .with_columns(
                pl.when(
                    pl.col('order').is_in(vrange)
                ).then(
                    pl.struct(['reference', 'distortion']).apply(
                        lambda x: func(
                            os.path.join(config.ref_dir, x['reference']),
                            os.path.join(config.dis_dir, 
                                         x['reference'].split('.')[0],
                                         x['distortion']),
                            args
                        ) 
                    ) 
                ).otherwise(
                    pl.col('metric') 
                ) 
                .alias('metric') 
            ) 
        ) 
    df = df.drop(['order','reference']) 
    df = df.rename({'distortion': 'name', 'metric':'mos'})
    return df 

def get_all_metrics(config):
    global METRICS
    ref_path = os.path.join(config.ref_dir, '*.ply')
    ref_objs = glob.glob(ref_path, recursive=True)

    references = []
    distortions = []
    for obj in ref_objs: 
        ref_base = os.path.basename(obj)
        ref_name = os.path.basename(obj).split('.')[0]
        dis_path = os.path.join(config.dis_dir, ref_name, '**/*.ply')
        dis_obj = glob.glob(dis_path, recursive=True)

        references += [ref_base for dis in dis_obj] 
        distortions += [os.path.basename(dis) for dis in dis_obj] 

    df = pl.DataFrame(zip(references, distortions), schema={'reference': pl.Utf8, 'distortion': pl.Utf8}) 

    for key in list(METRICS.keys()): 
        # Get the lower bound, upper bound and function from the metrics 
        values = METRICS[key]
        # Get the function that estimates the quality error[last pos]
        fvalues = values[2].split("/") 
        fname = fvalues[0]
        args = None 
        if len(fvalues) > 1: 
            args = fvalues[1]
        func = getattr(pcmm, fname) 
        # Iterate throught all the intensities 
        df = (
            df 
            .with_columns( 
                pl.struct(['reference', 'distortion'])
                .apply( 
                    lambda x: func(
                        os.path.join(config.ref_dir, x['reference']),
                        os.path.join(config.dis_dir, 
                                     x['reference'].split('.')[0],
                                     x['distortion']),
                        args
                    ) 
                )
                .alias(fname) 
            ) 
        )
        break 

    print(df) 

def main(config: dict = None): 
    df = None 
    if not config.all: 
        df = get_correct_metric(config)
    else: 
        df = get_all_metrics(config) 
    
    # Save the metrics in the distortion directory 
    if df is not None: 
        arr_dir = os.path.join(config.dis_dir, config.out_name) 
        df.write_csv(arr_dir) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('-i', '--ref-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/STDRemoved/') 

    parser.add_argument('-o', '--dis-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/Distortions/') 

    parser.add_argument('-n', '--out-name', type=str, default='metrics.csv') 

    parser.add_argument('-a', '--all', action='store_true') 
    parser.add_argument('-d', action='store_true') 

    config = parser.parse_args()
    if config.d:  
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(3)) 
    else: 
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0)) 

    main(config)
