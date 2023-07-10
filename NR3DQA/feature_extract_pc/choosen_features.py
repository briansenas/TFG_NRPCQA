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

    # pca2 = [i for item in get_geometry_nss_param(pca2) for i in item]
    # omni = [i for item in get_geometry_nss_param(omnivariance) for i in item]
    # egen = [i for item in get_geometry_nss_param(eigenentropy) for i in item]
    vert = [item for item in estimate_basic_param(verticality) ]
    pca2 = [item for item in estimate_basic_param(pca2) ]
    omni = [item for item in estimate_basic_param(omnivariance) ]
    egen = [item for item in estimate_basic_param(eigenentropy) ]

    return [name, *pca2, *omni, *egen, *vert]


def main(config: dict) -> None: 
    path = os.path.join(config.input_dir, '**/*.ply')
    objs = sorted(glob.glob(path, recursive=True))
    my_names = pl.read_csv('./features/WPC.csv')
    my_names = my_names.sort('name')
    my_names.write_csv('./features/WPC.csv') 
    my_names = my_names.select('name')['name'].to_list()
    other_names = [n.split('/')[-1] for n in objs] 
    names = set(other_names) - set(my_names) 
    remaining = []
    for x in objs: 
        if x.split('/')[-1] in names: 
            remaining += [x] 
    objs = remaining
    until = None
    if __debug__: 
        until = 1
    feat = []
    with open(config.output_dir, mode="a") as f:
        params = ["mean","std","entropy"]
        gdnames = ['pca2', 'omni', 'egen' ] 
        names = [g + "_" + p for g in gdnames for p in params]
        names.insert(0, "name") 
        names += ['vert_mean', 'vert_std', 'vert_entropy'] 
        # f.write(','.join(names))
        f.write('\n')
        with mp.get_context('spawn').Pool(4) as pool:  
            for res in tqdm(pool.imap_unordered(get_choosen_features, objs[:until]), 
                            total=len(objs[:until])):
                f.write(','.join([str(x) for x in res]))
                f.write('\n')

    if __debug__: 
        with pl.Config(tbl_rows=50):
            print(pl.read_csv(config.output_dir))


if __name__ == '__main__': 
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='./models'
                        )
    parser.add_argument('--output-dir', type=str, default="./features/choosen_features.csv") 

    config = parser.parse_args()

    main(config) 
