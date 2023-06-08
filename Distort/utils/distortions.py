import open3d as o3d
import numpy as np
import math
import argparse
from functools import partial 
import random 
import subprocess 

np.random.seed(140421) 
random.seed(140421) 

def get_basepath(): 
    basepath = __file__ 
    basepath = basepath[:basepath.rfind('/')+1]
    return basepath

def copy_helper(
    pcd: o3d.geometry.PointCloud,
): 
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = pcd.points
    new_pcd.normals = pcd.normals  
    new_pcd.colors = pcd.colors 
    return new_pcd 

""" 
The values for this type of distortions according to 
the paper are: 1.1 til 1.7 (vary the decimal by 0.1) 

{ 'contrast_change': [1.1,1.2,1.3,1.4,1.5,1.6,1.7] } 
""" 
def contrast_change( 
    pcd: o3d.geometry.PointCloud, 
    intensity_rate: float = 0.1 
): 
    intensity_rate = float(intensity_rate) 
   
    colors = np.asarray(pcd.colors) 
    colors = np.power(colors,intensity_rate) 

    new_pcd = copy_helper(pcd) 
    new_pcd.colors = o3d.utility.Vector3dVector(colors) 

    return new_pcd 

def _gamma_noise(shape, B, a):
    c = 1/a
    Ei = np.random.exponential(scale=c, size=(B, *shape))
    return np.sum(Ei, axis=0) / (3*255)

"""
For this distortion we must apply it with 
the following list of values: 

{ 'gamma_noise': [0.1,0.08,0.07, 0.06, 0.05, 0.04, 0.03] } 
"""
def gamma_noise(
    pcd: o3d.geometry.PointCloud(),
    a: float = 0.1,
    B: float = 3
): 
    colors = np.asarray(pcd.colors) 
    gnoise = _gamma_noise(colors.shape, B, a)
    gammad = copy_helper(pcd) 
    gammad.colors = o3d.utility.Vector3dVector(colors + gnoise)
    return gammad
""" 
For this one we must apply a percentage of the bounding box size: 
{ 'gaussian_geometric_shift': [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1] / 100 }  
""" 
def gaussian_geometric_shift(
    pcd: o3d.geometry.PointCloud(),
    intensity: float = 0.1 
): 
    np.random.seed(140421) 
    random.seed(140421) 
    intensity = float(intensity) 
    assert intensity >= 0.0 and intensity <= 1.0
    # Compute the bounding box of the point cloud
    xyz = np.asarray(pcd.points) 
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    # Define the standard deviation of the Gaussian distribution
    stdev = [intensity * extent[i] for i in range(3)]

    # Generate the random Gaussian shifts for each point
    num_points = len(xyz)
    shifts = np.random.normal(scale=stdev, size=(num_points, 3))

    # Apply the shifts to the point cloud
    shifted = copy_helper(pcd)
    shifted.points = o3d.utility.Vector3dVector(xyz + shifts) 

    return shifted 

""" 
{'additive_white_gaussian': [13, 11, 9, 7, 5, 3] } 
"""                     
def additive_white_gaussian(
    pcd: o3d.geometry.PointCloud, 
    target_snr: int = 13
): 

    target_snr = int(target_snr) 
    assert target_snr >= 0 and target_snr <= 100
    # Convert the point cloud to a numpy array
    colors = np.asarray(pcd.colors)
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(pcd.colors)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), size=colors.shape)
    # Apply the Additive White Gaussian blur
    awgd = copy_helper(pcd) 
    awgd.colors = o3d.utility.Vector3dVector(colors + noise_volts)
    return awgd

""" 
{'salt_pepper_noise': [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30] } 
""" 
def salt_pepper_noise(
    pcd: o3d.geometry.PointCloud,
    intensity_rate: float = 0.05
):
    intensity_rate = float(intensity_rate) 
    assert intensity_rate >= 0.0 and intensity_rate <= 1.0
    # Get point cloud coordinates
    colors = np.asarray(pcd.colors).copy() 

    # Get number of points in the point cloud
    num_points = colors.shape[0]

    # Create a mask for selecting random points
    noise_mask = np.random.choice([0, 1, 2], size=num_points, p=[intensity_rate, intensity_rate, 1 - 2*intensity_rate])

    # Add noise to point cloud
    colors[noise_mask == 0] = [0,0,0]
    colors[noise_mask == 1] = [1,1,1]

    # Update point cloud with noisy coordinates
    new_pcd = o3d.geometry.PointCloud() 
    new_pcd = copy_helper(pcd) 
    new_pcd.colors = o3d.utility.Vector3dVector(colors)

    return new_pcd

""" 
{ 'downsample_point_cloud' : [0.15,0.30,0.45,0.60,0.70,0.80,0.90]} 
""" 
def downsample_point_cloud(
    point_cloud: o3d.geometry.PointCloud,
    percentage: float = 0.2
):
    percentage = float(percentage) 
    percentage = 1 - percentage  
    assert percentage >= 0.0 and percentage <= 1.0 
    # Downsample the point cloud using voxel grid downsampling.
    downsampled_point_cloud = point_cloud.random_down_sample(percentage)
    # Remove any points that were not selected in the voxel grid downsampling.
    downsampled_point_cloud = downsampled_point_cloud.crop(point_cloud.get_axis_aligned_bounding_box())
    # Return the downsampled point cloud.
    return downsampled_point_cloud

def generate_bounding_box(
    xyz: np.ndarray, 
    extent: np.ndarray
):
    min_coords = xyz - extent 
    max_coords = xyz + extent 
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coords, max_bound=max_coords)
    
    return bbox

def get_bbox_patch(
    pcd: o3d.geometry.PointCloud, 
): 
    # Get the bounding box of the point cloud 
    bbox = pcd.get_axis_aligned_bounding_box() 
    # Get maximuum length on all axis 
    extent = bbox.get_extent() 
    # Size of the path to be selected 
    rextent = extent * 0.2
    # Select a random point  
    xyz = np.asarray(pcd.points).copy() 
    xyz_ =  xyz[np.random.choice(xyz.shape[0])]
    # Get the bounding box surrounding the selected point: 
    sel_bbox = generate_bounding_box(xyz_, rextent/2) 
    # Get the indices of the points that are within bounds of the new bbox 
    sel_points_indices = sel_bbox.get_point_indices_within_bounding_box(pcd.points)
    # Now those points should be offset 5% of the maximum side length of the bounding box 
    max_side_length = np.max(extent) 
    to_offset = 0.01 * max_side_length

    return sel_bbox, sel_points_indices, xyz, to_offset

def local_offset(
    pcd: o3d.geometry.PointCloud,
    level: int = 1,
    seed: bool = True,
): 
    if seed: 
        np.random.seed(140421) 
        random.seed(140421) 

    level = int(level) 
    if level == 0: 
        return pcd

    sel_bbox, sel_points_indices, xyz, to_offset = get_bbox_patch(pcd)

    xyz[sel_points_indices] += to_offset 

    obj = copy_helper(pcd) 
    obj.points = o3d.utility.Vector3dVector(xyz) 
    obj = local_offset(obj, level-1, False) 
    return obj 

def local_rotation(
    pcd: o3d.geometry.PointCloud,
    level: int = 1,
    rotation: float = 15, 
    seed: bool = True,
): 
    if seed: 
        np.random.seed(140421) 
        random.seed(140421) 
    level = int(level) 
    rotation = float(rotation) 
    if level == 0: 
        return pcd 

    sel_bbox, sel_points_indices, xyz, _ = get_bbox_patch(pcd)

    # Convert the rotation angle to radians
    rotation_angle_rad = np.radians(rotation)
    # Generate a partial point cloud from the selected points 
    partial_cloud = o3d.geometry.PointCloud() 
    partial_cloud.points = o3d.utility.Vector3dVector(xyz[sel_points_indices].copy()) 
    # Get the centroid of the partial point cloud  for the rotation
    center = partial_cloud.get_center() 
    # Get the rotation matrix for the x axis 
    R = partial_cloud.get_rotation_matrix_from_xyz((rotation_angle_rad, 0, 0))
    # Rotate the partial point_cloud with respect to its centroid  
    partial_cloud.rotate(R, center=center)
    # Re-attach the partial point cloud to the original point cloud 
    xyz[sel_points_indices] = np.asarray(partial_cloud.points) 
    # Use my copy_helper to create the new point cloud 
    obj = copy_helper(pcd) 
    obj.points = o3d.utility.Vector3dVector(xyz) 
    # We now recursively, according to the distortion level, get a new path 
    # and rotate with increasing rotation angle 
    obj = local_rotation(obj, level-1, rotation, False) 
    return obj

def octree_compression(
    input_dir: str,  
    output_dir: str,  
    octree_resolution: float 
): 
    octree_resolution = float(octree_resolution) 
    cmd = f'{get_basepath()}../octree/build/point_cloud_compression '
    cmd += f'{input_dir} {octree_resolution} {output_dir} ' 
    # execute the command and capture the output
    output = subprocess.check_output(cmd, shell=True)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visualize', type=bool, default=True)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-f', '--func', type=str, default='downsample_point_cloud') 
    parser.add_argument('-l','--args', nargs='+', required=False)
    parser.add_argument('--test-values', action='store_true') 
    parser.add_argument('--N', type=int, default=2000) 
    parser.add_argument('-i', '--input_dir', type=str, default=f"/home/briansenas/Desktop/PCQA-Databases/SJTU-PCQA/SJTU-PCQA/reference/hhi.ply") 
    parser.add_argument('-p', '--is-path-arg', action='store_true') 


    config = parser.parse_args()

    if config.args is None: 
        config.args = []
    
    intervalos_ = { 
        'salt_pepper_noise': [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30], 
        'downsample_point_cloud' : [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90],
        'contrast_change': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        'gamma_noise': [0.1,0.08, 0.07, 0.06, 0.05, 0.04, 0.03], 
        'gaussian_geometric_shift': [0.001, 0.0025, 0.004, 0.0055, 0.007, 0.0085, 1], 
        'additive_white_gaussian': [13, 11, 9, 7, 5, 3] 
    }
    if config.debug: 
        N = config.N
        if not config.is_path_arg: 
            pcd = o3d.io.read_point_cloud(
                config.input_dir
            )
        else: 
            pcd = config.input_dir
        try: 
            if not config.test_values: 
                obj = locals()[config.func](pcd, *config.args)

                if config.visualize and obj: 
                    o3d.visualization.draw_geometries([pcd]) 
                    o3d.visualization.draw_geometries([obj]) 
            if config.test_values: 
                o3d.visualization.draw_geometries([pcd]) 
                _ranges = intervalos_[config.func] 
                for r in _ranges: 
                    obj = locals()[config.func](pcd, r) 
                    o3d.visualization.draw_geometries([obj]) 

        except KeyError as e: 
            print("Available Functions: " + list(intervalos_.keys())) 
        except BaseException as e: 
            print(repr(e)) 
