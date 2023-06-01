import os
import argparse
import open3d as o3d
import open3d.core as o3c
import numpy as np
import glob 
import warnings 
from tqdm import tqdm 
import multiprocessing as mp
import functools 
import struct 
import utils.distortions as mypcd
import utils.metrics as pcmm  

DISTORTIONS = { 
    'downsample_point_cloud' : [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90], # 7 -> 13
    'local_offset': [1,2,3,4,5,6,7], # 14 -> 20 
    'local_rotation': [1,2,3,4,5,6,7], # 21 -> 27
    'gaussian_geometric_shift': [0.001, 0.0025, 0.004, 0.0055, 0.007, 0.0085, 0.01], # 28 -> 34
}

OCTREE = {
    'octree_compression': [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], # 0 -> 6
}

def write_pointcloud(filename,xyz, nxyz, rgb=None):

    """ creates a .pkl file of the point clouds generated
    """
    assert xyz.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb is None:
        rgb = np.ones(xyz.shape).astype(np.uint8)*255
    assert xyz.shape == rgb.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header  of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    if nxyz is not None: 
        fid.write(bytes('property float nx\n', 'utf-8'))
        fid.write(bytes('property float ny\n', 'utf-8'))
        fid.write(bytes('property float nz\n', 'utf-8'))

    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    if nxyz is not None: 
        for i in range(xyz.shape[0]):
            fid.write(bytearray(struct.pack("ffffffccc",xyz[i,0],xyz[i,1],xyz[i,2],
                                            nxyz[i,0], nxyz[i,1], nxyz[i,2], 
                                            rgb[i,0].tobytes(), rgb[i,1].tobytes(),
                                            rgb[i,2].tobytes()
                                            )
                                )
                      )
    else: 
        for i in range(xyz.shape[0]):
            fid.write(bytearray(struct.pack("fffccc",xyz[i,0],xyz[i,1],xyz[i,2],
                                            rgb[i,0].tobytes(), rgb[i,1].tobytes(),
                                            rgb[i,2].tobytes()
                                            )
                                )
                      )
    fid.close()


def estimate_point_spacing(point_cloud):
    point_cloud = np.asarray(point_cloud.points)
    # Compute the point density
    point_density = point_cloud.shape[0] / np.prod(np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0))

    # Estimate the average point spacing
    point_spacing = (1 / point_density) ** (1/3)

    return point_spacing

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def process_object(
    obj_path: str, 
    out_path: str, 
    visualize: bool = False,  
    outliers: bool = False
):  
    global DISTORTIONS, OCTREE
    #region Preprocessing: Remove outliers to/and center point cloud 
    outname = obj_path[obj_path.rfind("/")+1:].split('.')[0]
    out_path = os.path.join(out_path, outname)
    if not os.path.exists(out_path): 
        generate_dir(out_path)
    cloud = pcmm.read_point_cloud(obj_path, outliers) 
    #endregion

    #region Create all the distortions of the given file
    i = 0

    # We do the octree distortion separately
    for key in tqdm(list(OCTREE.keys()), leave=False):  
        func = getattr(mypcd, key)
        for value in tqdm(OCTREE[key], leave=False) : 
            output = os.path.join(out_path, outname + "_" + str(i) + ".ply") 
            obj = func(obj_path, output, value)
            i += 1

    # obj = None
    # for key in tqdm(list(DISTORTIONS.keys()), leave=False):  
    #     func = getattr(mypcd, key)
    #     for value in tqdm(DISTORTIONS[key], leave=False) : 
    #         obj = func(cloud, value)
    #         output = os.path.join(out_path, outname + "_" + str(i) + ".ply") 
    #         points = np.asarray(obj.points,dtype=np.float64) 
    #         normals = np.asarray(obj.normals,dtype=np.float64) 
    #         colors = np.asarray(obj.colors) * 255
    #         colors = np.asarray(colors,dtype=np.uint8) 
    #         write_pointcloud(output, points, normals, colors)
    #         i += 1

        # if visualize: 
        #     o3d.visualization.draw_geometries([obj]) 
    #endregion 

    # Overwrite the orignal one with the gray color 
    # points = np.asarray(cloud.points,dtype=np.float64) 
    # normals = np.asarray(cloud.normals,dtype=np.float64) 
    # colors = np.asarray(cloud.colors) * 255
    # colors = np.asarray(colors,dtype=np.uint8) 
    # write_pointcloud(obj_path, points, normals, colors) 

def main(configs: dict = None):
    path = os.path.join(config.input_dir, '*.ply')
    objs = glob.glob(path, recursive=True)

    until = None
    if __debug__: 
        until = 1
    func = functools.partial(process_object, out_path=config.output_dir, visualize=config.d, outliers=config.outliers) 
    with mp.get_context('forkserver').Pool(processes=mp.cpu_count()) as pool:  
        feat = []
        for res in tqdm(pool.imap_unordered(func, objs[:until]), 
                        total=len(objs[:until])):
            continue 
    # for obj in objs: 
    #     process_object(obj, config.output_dir, config.d, config.outliers) 
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('-i', '--input-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/STDRemoved/') 

    parser.add_argument('-o', '--output-dir', type=str, 
                        default='/home/briansenas/Desktop/PCQA-Databases/OurData/Distortions/') 
    parser.add_argument('-d', action='store_true') 
    parser.add_argument('-s', '--outliers', action='store_true') 
    config = parser.parse_args()
    if config.d:  
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(3)) 
    else: 
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0)) 

    main(config)
