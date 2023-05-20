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
from nss_functions import Entropy

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def get_new_features(
    cloud 
) -> pl.DataFrame: 
    name = os.path.basename(cloud)
    df = pl.DataFrame({'name': name})

    pcd = o3d.io.read_point_cloud(cloud) 
    pcd.estimate_normals() 
    pcd.estimate_covariances()
    pcd.compute_convex_hull()
    pcd.orient_normals_consistent_tangent_plane(10)
    covariance_matrix = pcd.covariances
    
    lambda_, eigvector = np.linalg.eig(covariance_matrix)
    
    eigen_value_sum = lambda_[:, 0] + lambda_[:, 1] + lambda_[:, 2] 
    # Omnivariance 
    pca1 = lambda_[:, 0] / eigen_value_sum[:]
    pca2 = lambda_[:, 1] / eigen_value_sum[:]
    surface_variation_factor = lambda_[:, 2] / eigen_value_sum[:]
    verticality = 1.0 - abs(eigvector[:, 2, 2])
    eigen_ratio = np.max(lambda_, axis=1) / np.min(lambda_, axis=1) 
    
    cloud = PyntCloud.from_file(cloud)
    #begin geometry projection
    k_neighbors = cloud.get_neighbors(k=30)
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    cloud.add_scalar_field("omnivariance", ev=ev)
    cloud.add_scalar_field("eigenentropy",ev=ev)
    omnivariance = cloud.points['omnivariance(31)'].to_numpy()
    eigenentropy = cloud.points['eigenentropy(31)'].to_numpy()

    generate_dir(config.save_npy) 
    save_dir = os.path.join(config.save_npy, name) 
    generate_dir(save_dir) 
    varnames = [i for i, k in locals().items() if isinstance(k, np.ndarray) 
                and np.array_equal(k, k_neighbors) and np.array_equal(k, lambda_)
                and np.array_equal(k, eigen_value_sum)]
    # Save variable based on the string name of the variable using synmbol
    for var in varnames:  
        np.save(os.path.join(save_dir, var + ".npy"), locals()[var])
    
    eigdf = pl.DataFrame({
        'pca1_mean': np.mean(pca1), 
        'pca2_mean': np.mean(pca2), 
        'svf_mean': np.mean(surface_variation_factor),
        'vert_mean': np.mean(verticality), 
        'vert_dsv' : np.std(verticality), 
        'Re_mean': np.mean(eigen_ratio),  
        'omni_mean': np.mean(omnivariance), 
        'eigenentropy_mean': np.mean(eigenentropy)
    }) 
    
    df = pl.concat([df, eigdf], how='horizontal') 
    
    # with o3d.utility.VerbosityContextManager(
    #         o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, _ = (
        o3d.geometry.TriangleMesh
            .create_from_point_cloud_poisson(pcd, depth=10, width=0,
                                             scale=20, linear_fit=True)
        )
    mesh.remove_degenerate_triangles()
    surface_area = mesh.get_surface_area() 
    volume = mesh.get_volume()
    # calculate compactness
    bbox = pcd.get_minimal_oriented_bounding_box()
    bbox_volume = bbox.volume() 
    compactness = volume /  bbox_volume
    # Compute density
    xyz = np.asarray(pcd.points) 
    density = xyz.shape[0]/ volume
    # Compute the uniformity
    uniformity = IsoScore.IsoScore(xyz.transpose())

    eigdf = pl.DataFrame({
        'surf_area': surface_area, 
        'compactness': compactness,
        'density': density, 
        'uniformity': uniformity, 
    })

    df = pl.concat([df, eigdf], how='horizontal') 
    
    return df 
def main(config: dict) -> None: 
    path = os.path.join(config.input_dir, '*.ply')
    objs = glob.glob(path, recursive=True)
    df = pl.DataFrame() 
    until = 1
    if __debug__: 
        until = None
    for obj in tqdm(objs[:until]): 
        df = pl.concat([df, get_new_features(obj)], how='vertical')
        break 

    if __debug__: 
        with pl.Config(tbl_rows=50):
            print(df)
    df.write_csv(config.output_dir)

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/SJTU-PCQA/SJTU-PCQA/reference/'
                        )
    parser.add_argument('--output-dir', type=str, default="./features/new_features.csv") 
    parser.add_argument('--save_npy', type=str, default='./rawnp')

    config = parser.parse_args()

    main(config) 
