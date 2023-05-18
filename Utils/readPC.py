import numpy as np
import os 
import argparse 
import open3d as o3d
import struct 
import glob 

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

def test_wpc(
    file_name: str, 
    output_name: str, 
    visualize: bool = True,
): 
    cloud = o3d.io.read_point_cloud(file_name) 
    cloud.estimate_normals() 
    cloud = cloud.farthest_point_down_sample(num_samples=int(0.90*len(cloud.points)))
    cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=256, std_ratio=2.0) 

    if not cloud.colors: 
        cloud.colors = o3d.utility.Vector3dVector(np.ones(shape=(len(cloud.points),3))*0.5)

    points = np.asarray(cloud.points,dtype=np.float32) 
    normals = np.asarray(cloud.normals,dtype=np.float32) 
    colors = np.asarray(cloud.colors) * 255
    colors = np.asarray(colors,dtype=np.uint8) 
    write_pointcloud(output_name, points, normals, colors) 

    if visualize: 
        cloud = o3d.io.read_point_cloud(output_name) 
        o3d.visualization.draw_geometries([cloud]) 



def convert_mesh(
    file_name: str,
    output_name: str ,
    N: int = 200000,
    visualize: bool = False
):
    if output_name is None: 
        output_name = file_name.split('.')[0] + ".ply"

    mesh = o3d.io.read_triangle_mesh(file_name, True, True)
    # Apply the materials to the mesh.
    cloud = mesh.sample_points_uniformly(N)
    cloud.estimate_normals() 
    colors = np.zeros(shape=np.asarray(cloud.points).shape)+ [.5,.5,.5]
    cloud.colors = o3d.utility.Vector3dVector(colors) 
    # cloud = o3d.io.read_point_cloud("./radiantobject.stl") # Read the point cloud
    if visualize: 
        o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud

    o3d.io.write_point_cloud(output_name, cloud)

def read_pc(file_name, *args): 
    cloud = o3d.io.read_point_cloud(file_name) 
    cloud.estimate_normals() 
    # cloud = cloud.farthest_point_down_sample(num_samples=int(0.90*len(cloud.points)))
    cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=32, std_ratio=5.0) 
    o3d.visualization.draw_geometries([cloud]) 

def read_all_pc(file_name, *args): 

    ref_path = os.path.join(config.input_obj, '*.ply')
    ref_objs = glob.glob(ref_path, recursive=True)

    for obj in ref_objs: 
        cloud = read_point_cloud(obj) 
        o3d.visualization.draw_geometries([cloud])


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
# Somehow It is not working lmao
def convert_obj_to_ply(file_name, output_file=None): 
    mesh = om.TriMesh()
    om.read_trimesh(file_name)
    if output_file is None: 
        output_file = file_name.split('.')[0] + ".ply"
    om.write_mesh(output_file, mesh)
    return output_file 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--func', type=str, default="convert_mesh")
    parser.add_argument('-i','--input-obj', type=str, default="Segmentation.obj")
    parser.add_argument('-o','--output', type=str, default=None)
    config = parser.parse_args()
    file_name = config.input_obj
    output_name = config.output
    locals()[config.func](file_name, output_name)
