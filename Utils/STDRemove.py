import argparse 
import os 
import glob 
import numpy as np
import copy
import open3d as o3d

def estimate_point_spacing(point_cloud):
    point_cloud = np.asarray(point_cloud.points)
    # Compute the point density
    point_density = point_cloud.shape[0] / np.prod(np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0))

    # Estimate the average point spacing
    point_spacing = (1 / point_density) ** (1/3)

    return point_spacing

def read_point_cloud(
    fileA: str = None,
) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(fileA) 
    cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=32, std_ratio=5.0) 
    if not cloud.normals or len(cloud.normals) <= 0: 
        cloud.estimate_normals(
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=estimate_point_spacing(cloud), 
                max_nn=128)
        )
   
    if not cloud.colors or len(cloud.colors) <= 0: 
        colors = np.zeros(shape=np.asarray(cloud.points).shape)+ [.5,.5,.5]
        cloud.colors = o3d.utility.Vector3dVector(colors) 
    
    return cloud 


def start_preprocess(config): 
    path = os.path.join(config.input_dir, '*.ply')
    objs = glob.glob(path, recursive=True)
    for obj in objs: 
        cloud = read_point_cloud(obj) 
        outname = obj[obj.rfind("/")+1:]
        outdir = os.path.join(config.output_dir, outname)
        o3d.io.write_point_cloud(outdir, cloud) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-dir', type=str, default="Segmentation.obj")
    parser.add_argument('-o','--output-dir', type=str, default="test_wpc.ply")
    config = parser.parse_args()
    start_preprocess(config) 
