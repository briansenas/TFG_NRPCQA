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
import pcmetrics_module as pcmm
from tqdm import tqdm 
#endregion imports 

#region GLOBALS 
# METRICS = { 
#     '0': [0, 6, "pcqm"],
#     '1': [7, 13, "run_pcc/mseF (p2point)"],
#     '2': [14, 20, "hpsnryuv"],
#     '3': [21, 27, "psnryuv"],
#     '4': [28, 34, "run_pcc/h. (p2plane)"],
#     '5': [35, 41, "pcqm"],
# }

METRICS = { 
    '0': [0, 6, "pcqm"], # Should divide by 10
    '1': [7, 13, "run_pcc/mseF (p2point)"], # SHould divide by the diagonal of the bbx
    '2': [14, 20, "psnryuv"],
    '3': [21, 27, "run_pcc/h. (p2plane)"],
    '4': [28, 34, "pcqm"], # Should divide by 10
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
    print(df) 

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
    # ref_path = os.path.join(config.ref_dir, '*.ply')
    # ref_objs = glob.glob(ref_path, recursive=True)
    # # numpy array of metrics 
    # pcname = []
    # metrics = []
    # # For each reference object 
    # for objs in tqdm(ref_objs): 
    #     ref_name = os.path.basename(objs).split('.')[0]
    #     # For each distortion type 
    #     for key in tqdm(list(METRICS.keys()), leave=False): 
    #         # Get the lower bound, upper bound and function from the metrics 
    #         values = METRICS[key]
    #         # Get the function that estimates the quality error[last pos]
    #         fvalues = values[2].split("/") 
    #         fname = fvalues[0]
    #         args = None 
    #         if len(fvalues) > 1: 
    #             args = fvalues[1]
    #         func = getattr(pcmm, fname) 
    #         # Iterate throught all the intensities 
    #         for i in tqdm(range(values[0], values[1]+1), leave=False): 
    #             dis_name = ref_name + f"_{i}.ply"
    #             dis_obj = os.path.join(config.dis_dir, ref_name, dis_name) 
    #             pcname.append(dis_name) 
    #             metrics.append(func(dis_obj, objs, args)) 
    #
    # # Create a flattened matrix of metrics 
    # df = pl.DataFrame(zip(pcname,metrics), schema={'name': pl.Utf8,'mos': pl.Float64}) 
    # return df 

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
    if config.correct_only: 
        df = get_correct_metric(config)
    else: 
        df = get_all_metrics(config) 
    
    # Save the metrics in the distortion directory 
    if df is not None: 
        arr_dir = os.path.join(config.dis_dir, "metrics.csv") 
        df.write_csv(arr_dir) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('-i', '--ref-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/STDRemoved/') 

    parser.add_argument('-o', '--dis-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/Distortions/') 

    parser.add_argument('-c', '--correct-only', action='store_true') 
    parser.add_argument('-d', action='store_true') 

    config = parser.parse_args()
    if config.d:  
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(3)) 
    else: 
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0)) 

    main(config)
