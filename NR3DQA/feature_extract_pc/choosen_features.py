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

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def get_new_features(
    cloud 
) -> pl.DataFrame: 
    name = os.path.basename(cloud)
    df = pl.DataFrame({'name': name})
    nn_ = 30
    epsilon = 1e-6
    pcd = o3d.io.read_point_cloud(cloud) 
    pcd.estimate_normals() 
    pcd.estimate_covariances()
    pcd.orient_normals_consistent_tangent_plane(10)
    xyz = np.asarray(pcd.points) 
    covariance_matrix = pcd.covariances
    lambda_, eigvector = np.linalg.eig(covariance_matrix)
    
    eigen_value_sum = lambda_[:, 0] + lambda_[:, 1] + lambda_[:, 2] + epsilon
    # Omnivariance 
    pca1 = lambda_[:, 0] / eigen_value_sum[:]
    pca2 = lambda_[:, 1] / eigen_value_sum[:]
    verticality = 1.0 - abs(eigvector[:, 2, 2])

    omnivariance = np.nan_to_num(
        pow(lambda_[:, 0] * lambda_[:, 1] * lambda_[:, 2] + epsilon, 1.0 / 3.0)
    )

    eigenentropy = np.nan_to_num(
        np.negative(
            lambda_[:,0] * np.log(lambda_[:, 0]+epsilon) 
                + lambda_[:, 1] * np.log(lambda_[:, 1]+epsilon) 
                + lambda_[:, 2] * np.log(lambda_[:, 2]+epsilon)
        )
    )
    
    pca2 = [i for item in get_geometry_nss_param(pca2) for i in item]
    omni = [i for item in get_geometry_nss_param(omnivariance) for i in item]
    egen = [i for item in get_geometry_nss_param(eigenentropy) for i in item]
    vert = [item for item in estimate_basic_param(verticality) ]

    # Compute the uniformity
    uniformity = IsoScore.IsoScore(xyz.transpose())

    dictdf = {} 
    params = ["mean","std","entropy","ggd1","ggd2","aggd1","aggd2","aggd3","aggd4","gamma1","gamma2"]
    gdnames = ['pca2', 'omni', 'egen' ]
    varlist = [pca2,  omni, egen]
    for i, var in enumerate(gdnames):
        for j, name in enumerate(params): 
            key_ = var + "_" + name 
            dictdf.update({key_: varlist[i][j]}) 
    dictdf.update({'vert_mean': vert[0], 'vert_std': vert[1], 'vert_entropy': vert[2]}) 
    dictdf.update({'uniformity': uniformity})
    df = pl.concat([df, pl.DataFrame(dictdf)], how='horizontal') 
    
    return df 


def main(config: dict) -> None: 
    path = os.path.join(config.input_dir, '*.ply')
    objs = glob.glob(path, recursive=True)
    df = pl.DataFrame() 
    until = None
    if __debug__: 
        until = 1
    for obj in tqdm(objs[:until]): 
        df = pl.concat([df, get_new_features(obj)], how='vertical')

    if __debug__: 
        with pl.Config(tbl_rows=50):
            print(df)
    df.write_csv(config.output_dir)


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='./models'
                        )
    parser.add_argument('--output-dir', type=str, default="./features/choosen_features.csv") 
    parser.add_argument('--save_npy', type=str, default='./rawnp')

    config = parser.parse_args()

    main(config) 
