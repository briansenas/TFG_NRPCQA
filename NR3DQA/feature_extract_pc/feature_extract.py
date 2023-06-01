import numpy as np
import pandas as pd
import os
import glob 
import argparse 
import numpy as np
import polars as pl 
from skimage import color
from nss_functions import * 
from pyntcloud import PyntCloud
import multiprocessing as mp
from tqdm import tqdm

NAMES = """l_mean,l_std,l_entropy,a_mean,a_std,a_entropy,b_mean,b_std,b_entropy,curvature_mean,curvature_std,curvature_entropy,curvature_ggd1,curvature_ggd2,curvature_aggd1,curvature_aggd2,curvature_aggd3,curvature_aggd4,curvature_gamma1,curvature_gamma2,anisotropy_mean,anisotropy_std,anisotropy_entropy,anisotropy_ggd1,anisotropy_ggd2,anisotropy_aggd1,anisotropy_aggd2,anisotropy_aggd3,anisotropy_aggd4,anisotropy_gamma1,anisotropy_gamma2,linearity_mean,linearity_std,linearity_entropy,linearity_ggd1,linearity_ggd2,linearity_aggd1,linearity_aggd2,linearity_aggd3,linearity_aggd4,linearity_gamma1,linearity_gamma2,planarity_mean,planarity_std,planarity_entropy,planarity_ggd1,planarity_ggd2,planarity_aggd1,planarity_aggd2,planarity_aggd3,planarity_aggd4,planarity_gamma1,planarity_gamma2,sphericity_mean,sphericity_std,sphericity_entropy,sphericity_ggd1,sphericity_ggd2,sphericity_aggd1,sphericity_aggd2,sphericity_aggd3,sphericity_aggd4,sphericity_gamma1,sphericity_gamma2""".split(',')

def get_feature_vector(objpath):  
    global NAMES 

    # load colored point cloud
    # print("Begin loading point cloud")
    cloud = PyntCloud.from_file(objpath)

    name = os.path.basename(objpath) 

    # begin geometry projection
    # print("Begin geometry feature extraction.")
    k_neighbors = cloud.get_neighbors(k=10)
    ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    cloud.add_scalar_field("curvature", ev=ev)
    cloud.add_scalar_field("anisotropy",ev=ev)
    cloud.add_scalar_field("linearity",ev=ev)
    cloud.add_scalar_field("planarity",ev=ev)
    cloud.add_scalar_field("sphericity",ev=ev)
    curvature = cloud.points['curvature(11)'].to_numpy()
    anisotropy = cloud.points['anisotropy(11)'].to_numpy()
    linearity = cloud.points['linearity(11)'].to_numpy()
    planarity = cloud.points['planarity(11)'].to_numpy()
    sphericity = cloud.points['sphericity(11)'].to_numpy()


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
    return df

def main(config:dict) -> None: 
    path = os.path.join(config.input_dir, '**/*.ply')
    objs = glob.glob(path, recursive=True)
    df = pl.DataFrame() 
    until = None
    if __debug__: 
        until = 1
    # for obj in tqdm(objs[:until]): 
    #     df = pl.concat([df, get_feature_vector(obj)], how='vertical')

    with mp.get_context('forkserver').Pool() as pool:  
        for res in tqdm(pool.imap_unordered(get_feature_vector, objs[:until:]), 
                        total=len(objs[:until])):
            df = pl.concat([df, res], how='vertical')

    #show the features
    if __debug__: 
        with pl.Config(tbl_rows=50):
            print(df)

    df.write_csv(config.output_dir) 
    
#demo
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/SJTU-PCQA/SJTU-PCQA/reference/'
                        )
    parser.add_argument('--output-dir', type=str, default="./features/features.csv") 

    config = parser.parse_args()

    main(config) 

