#!/usr/bin/env python
# coding: utf-8
import os
import glob 
import argparse 
import numpy as np
import polars as pl 
import open3d as o3d
# @TODO: Make this parallel
import multiprocessing as mp
from pyntcloud import PyntCloud
from tqdm import tqdm 
from scipy import stats
from IsoScore import IsoScore
from nss_functions import get_geometry_nss_param, estimate_basic_param
import warnings 
import functools

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def get_choosen_features(
    cloud: str, 
) -> pl.DataFrame: 
    name = os.path.basename(cloud)
    print(name)
    # .split('.')[0].replace('_','')
    nn_ = 10

    cloud = PyntCloud.from_file(cloud)

    # begin geometry projection
    # print("Begin geometry feature extraction.")
    k_neighbors = cloud.get_neighbors(k=nn_)
    ev = cloud.add_scalar_field("eigen_decomposition",k_neighbors=k_neighbors)
    cloud.add_scalar_field("omnivariance", ev=ev)
    cloud.add_scalar_field("eigenentropy",ev=ev)
    cloud.add_scalar_field("eigen_sum",ev=ev)
    omnivariance = cloud.points[f'omnivariance({nn_+1})'].to_numpy() 
    eigenentropy = cloud.points[f'eigenentropy({nn_+1})'].to_numpy() 
    eigen_sum = cloud.points[f'eigen_sum({nn_+1})'].to_numpy() 
    eigvec_height = cloud.points[f'ev3_z({nn_+1})'].to_numpy()
    ev1 = cloud.points[f'e1({nn_+1})'].to_numpy()

    # Omnivariance 
    pca2 = ev1[:] / eigen_sum[:]
    verticality = 1.0 - abs(eigvec_height[:]) 

    pca2 = [i for item in get_geometry_nss_param(pca2) for i in item]
    omni = [i for item in get_geometry_nss_param(omnivariance) for i in item]
    egen = [i for item in get_geometry_nss_param(eigenentropy) for i in item]
    vert = [item for item in estimate_basic_param(verticality) ]

    # Compute the uniformity
    xyz = cloud.points.to_numpy()
    
    return [name, *pca2, *omni, *egen, *vert]


def main(config: dict) -> None: 
    path = os.path.join(config.input_dir, '**/*.ply')
    objs = glob.glob(path, recursive=True)
    until = None
    if __debug__: 
        until = 1
    feat = []
    with mp.get_context('forkserver').Pool() as pool:  
        for res in tqdm(pool.imap_unordered(get_choosen_features, objs[:until:]), 
                        total=len(objs[:until])):
            feat.append(res) 

    features = [f for f in feat] 
    features = np.array(features) 
    df = pl.DataFrame() 
    params = ["mean","std","entropy","ggd1","ggd2","aggd1","aggd2","aggd3","aggd4","gamma1","gamma2"]
    gdnames = ['pca2', 'omni', 'egen' ]
    names = [g + "_" + p for g in gdnames for p in params]
    names.insert(0, "name") 
    names += ['vert_mean', 'vert_std', 'vert_entropy'] 
    dictdf = {}
    for i, name in enumerate(names): 
        dictdf.update({name: features[:, i]}) 
    df = pl.DataFrame(dictdf) 
    if __debug__: 
        with pl.Config(tbl_rows=50):
            print(df)
    df.write_csv(config.output_dir)


if __name__ == '__main__': 
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='./models'
                        )
    parser.add_argument('--output-dir', type=str, default="./features/choosen_features.csv") 

    config = parser.parse_args()

    main(config) 
