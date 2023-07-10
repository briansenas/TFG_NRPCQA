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
from nss_functions import * 
from skimage import color
import warnings 
import functools

NAMES = """l_mean,l_std,l_entropy,a_mean,a_std,a_entropy,b_mean,b_std,b_entropy,curvature_mean,curvature_std,curvature_entropy,curvature_ggd1,curvature_ggd2,curvature_aggd1,curvature_aggd2,curvature_aggd3,curvature_aggd4,curvature_gamma1,curvature_gamma2,anisotropy_mean,anisotropy_std,anisotropy_entropy,anisotropy_ggd1,anisotropy_ggd2,anisotropy_aggd1,anisotropy_aggd2,anisotropy_aggd3,anisotropy_aggd4,anisotropy_gamma1,anisotropy_gamma2,linearity_mean,linearity_std,linearity_entropy,linearity_ggd1,linearity_ggd2,linearity_aggd1,linearity_aggd2,linearity_aggd3,linearity_aggd4,linearity_gamma1,linearity_gamma2,planarity_mean,planarity_std,planarity_entropy,planarity_ggd1,planarity_ggd2,planarity_aggd1,planarity_aggd2,planarity_aggd3,planarity_aggd4,planarity_gamma1,planarity_gamma2,sphericity_mean,sphericity_std,sphericity_entropy,sphericity_ggd1,sphericity_ggd2,sphericity_aggd1,sphericity_aggd2,sphericity_aggd3,sphericity_aggd4,sphericity_gamma1,sphericity_gamma2""".split(',')


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_feature_vector(
    cloud: str, 
) -> pl.DataFrame: 
    global NAMES 
    name = os.path.basename(cloud)
    # .split('.')[0].replace('_','')
    nn_ = 10

    cloud = PyntCloud.from_file(cloud)

    # begin geometry projection
    # print("Begin geometry feature extraction.")
    k_neighbors = cloud.get_neighbors(k=nn_)
    ev = cloud.add_scalar_field("eigen_decomposition",k_neighbors=k_neighbors)
    cloud.add_scalar_field("curvature", ev=ev)
    cloud.add_scalar_field("anisotropy",ev=ev)
    cloud.add_scalar_field("linearity",ev=ev)
    cloud.add_scalar_field("planarity",ev=ev)
    cloud.add_scalar_field("sphericity",ev=ev)
    curvature = cloud.points[f'curvature({nn_+1})'].to_numpy()
    anisotropy = cloud.points[f'anisotropy({nn_+1})'].to_numpy()
    linearity = cloud.points[f'linearity({nn_+1})'].to_numpy()
    planarity = cloud.points[f'planarity({nn_+1})'].to_numpy()
    sphericity = cloud.points[f'sphericity({nn_+1})'].to_numpy()

    # begin color projection
    # print("Begin color feature extraction.")
    rgb_color = cloud.points[['red','green','blue']].to_numpy()/255
    lab_color = color.rgb2lab(rgb_color)
    l = lab_color[:,0]
    a = lab_color[:,1]
    b = lab_color[:,2]

    # print("Begin NSS parameters estimation.")
    # compute nss parameters
    nss_params = []
    # compute color nss features
    for tmp in [l,a,b]:
      params = get_color_nss_param(tmp)
      #flatten the feature vector
      nss_params = nss_params + [i for item in params for i in item]
    # compute geomerty nss features
    for tmp in [curvature,anisotropy,linearity,planarity,sphericity]:
      params = get_geometry_nss_param(tmp)
      #flatten the feature vector
      nss_params = nss_params + [i for item in params for i in item]

    df = pl.DataFrame({'value': nss_params}) 
    df = df.transpose(include_header=False, column_names=NAMES)
    df = df.with_columns(pl.lit(name).alias('name'))

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


    params = ["mean","std","entropy"]
    gdnames = ['pca2', 'omni', 'egen' ]
    names = [g + "_" + p for g in gdnames for p in params]
    names += ['vert_mean', 'vert_std', 'vert_entropy'] 

    features =  np.asarray([*pca2, *omni, *egen, *vert])
    for i, name in enumerate(names): 
        df = df.with_columns(pl.lit(features[i]).alias(name)) 

    return df 


def main(config: dict) -> None: 
    path = os.path.join(config.input_dir, '**/*.ply')
    objs = glob.glob(path, recursive=True)
    df = pl.DataFrame()
    until = None
    if __debug__: 
        until = 1
    cont = 0
    with mp.get_context('forkserver').Pool() as pool:  
        with open("out.csv", mode="ab") as f:
            for res in tqdm(pool.imap_unordered(get_feature_vector, objs[:until:]), 
                            total=len(objs[:until])):
                # Continuous write_to_csv
                if cont == 0 and not config.continue: 
                    res.write_csv(f)
                else: 
                    res.write_csv(f, has_header=False) 
                # df = pl.concat([df, res], how='vertical')

    #show the features
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
    parser.add_argument('--continue', '-c', action='store_true') 

    config = parser.parse_args()

    main(config) 
