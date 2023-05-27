import os
import re 
import argparse
import subprocess 
import PCDistortions as mypcd
import open3d as o3d
import numpy as np
import glob 
import warnings 
from tqdm import tqdm 
import multiprocessing
import functools 
from scipy.optimize import curve_fit
from pydantic.typing import Union

def get_basepath(): 
    basepath = __file__ 
    basepath = basepath[:basepath.rfind('/')+1]
    return basepath

#region auxilizar_functions
def estimate_point_spacing(point_cloud):
    point_cloud = np.asarray(point_cloud.points)
    # Compute the point density
    point_density = point_cloud.shape[0] / np.prod(np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0))

    # Estimate the average point spacing
    point_spacing = (1 / point_density) ** (1/3)

    return point_spacing

def read_point_cloud(
    fileA: str = None,
    outliers: bool = False, 
) -> o3d.geometry.PointCloud:
    cloud = o3d.io.read_point_cloud(fileA) 
    if outliers: 
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
 

def rgb2yuv(rgb):
    # Convert the RGB color information to YUV
    maxi_ = np.amax(rgb) 
    rgb /= maxi_ 
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    yuv = np.column_stack((y, u, v))
    return yuv

def hausdorff_distance(
    fileA: str = None, 
    fileB: str = None,
    *args, 
) -> float :
    pcd1 = read_point_cloud(fileA) 
    pcd2 = read_point_cloud(fileB) 
    # Convert point clouds to numpy arrays
    distances = pcd1.compute_point_cloud_distance(pcd2)
    hausdorff_distance = np.max(distances) 

    return hausdorff_distance 
""" 
./pcqm reference_objet.ply registered_object.ply -r 0.004 -knn 20 -rx 2.0 -fq
""" 
def pcqm(
    fileA: str = None, 
    fileB: str = None,
    *args
) -> float: 
    # command to run the PCC quality measurement software
    cmd = f'{get_basepath()}../Softwares/pcqm {fileA} {fileB} -r 0.004 -knn 20 -rx 2.0 -fq'
    # execute the command and capture the output
    output = subprocess.check_output(cmd, shell=True)
    # decode the byte string output to string and split into lines
    output_lines = output.decode('utf-8').split('\n')
    # extract the final symmetric metrics from the output
    final = output_lines[-3:-2]

    return float(final[0].split(':')[1])

""" 
./pc_error_d --fileA=reference/redandblack.ply --fileB=distortion/redandblack/redandblack_0.ply --hausdorff=1 --nbThreads=3S
valid_keys = ['mseF (p2point)', 'mseF, PSNR (p2point)', 'mseF (p2plane)', 
              'mseF,PSNR (p2plane)', 'h. (p2point)', 'h.,PSNR (p2point)', 
              'h. (p2plane)', 'h.,PSNR (p2plane)' ] 
""" 
def run_pcc(
    fileA: str = None,
    fileB: str = None,
    find_key: str = ''
    ) -> Union[list, float]:
    # command to run the PCC quality measurement software
    cmd = f'{get_basepath()}../Softwares/pc_error_d --fileA={fileA} --fileB={fileB} --hausdorff=1 --nbThreads=32'
    # execute the command and capture the output
    output = subprocess.check_output(cmd, shell=True)
    # decode the byte string output to string and split into lines
    output_lines = output.decode('utf-8').split('\n')
    # extract the final symmetric metrics from the output
    final_metrics = output_lines[-10:-2]
    # create a dictionary to store the final symmetric metrics
    metrics = {}
    for line in final_metrics:
        key, value = line.split(':')
        key = re.sub(' +', ' ', key)
        metrics.update({key.strip():float(value.strip())})

    res = []
    if find_key: 
        if not isinstance(find_key, list): 
            find_key = [find_key] 

        for key in find_key:  
            res.append(metrics[key]) 
    else: 
        res = metrics 
    if len(res) == 1: 
        return res[0]
    return res 

def psnryuv(
    fileA: str = None, 
    fileB: str = None, 
    *args,
) -> float: 
    pcd1 = read_point_cloud(fileA) 
    pcd2 = read_point_cloud(fileB) 

    color1 = np.asarray(pcd1.colors) * 255
    color2 = np.asarray(pcd2.colors) * 255
    c1 = rgb2yuv(color1) 
    c2 = rgb2yuv(color2)

    if c1.shape[0] != c2.shape[0]:  
        mx = max(c1.shape[0], c2.shape[0])
        cp1 = np.zeros(shape=(mx,c1.shape[1]))
        cp2 = np.zeros(shape=(mx,c2.shape[1])) 
        # We must pad the color vector to be same size  
        cp1[:c1.shape[0]] = cp1[:c1.shape[0]] + c1
        cp2[:c2.shape[0]] = cp2[:c2.shape[0]] + c2 
    else: 
        cp1 = c1 
        cp2 = c2
    
    # Calculate the PSNR
    mse = np.mean((cp2 - cp1) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = -(10 * np.log10(
                          ( (255 - 1)**2 ) / np.sqrt(mse) 
                          )
            )
    return psnr 

def hpsnryuv( 
    fileA: str = None, 
    fileB: str = None,
    *args
) -> float: 

    psnr = psnryuv(fileA, fileB) 

    # Compute Hausdorff distance
    hausdorff_dist = hausdorff_distance(fileA, fileB) 

    return hausdorff_dist * psnr 

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--func', type=str, default='hpsnryuv') 
    parser.add_argument('-a', '--fileA', type=str, default='/home/briansenas/Desktop/PCQA-Databases/SJTU-PCQA/SJTU-PCQA/reference/hhi.ply') 
    parser.add_argument('-b', '--fileB', type=str, default='/home/briansenas/Desktop/PCQA-Databases/SJTU-PCQA/SJTU-PCQA/reference/hhi.ply') 
    parser.add_argument('-l','--args', nargs='+', required=False)
    config = parser.parse_args()

    try: 
        if config.args: 
            value = locals()[config.func](config.fileA, config.fileB, *config.args)
        else: 
            value = locals()[config.func](config.fileA, config.fileB)

        print(value)

    except KeyError as e: 
        print("Available Functions: " + locals()) 
    except BaseException as e: 
        print(repr(e)) 
