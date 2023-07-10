# examples/Python/Advanced/interactive_visualization.py
import argparse 
import os 
import glob 
import numpy as np
import copy
import open3d as o3d


def demo_crop_geometry(obj_path):
    pcd = o3d.io.read_point_cloud(obj_path)
    # o3d.visualization.draw_geometries_with_editing([pcd])
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    print(dir(vis)) 
    if not pcd.colors or len(pcd.colors) <= 0: 
        colors = np.zeros(shape=np.asarray(pcd.points).shape)+ [.5,.5,.5]
        pcd.colors = o3d.utility.Vector3dVector(colors) 
    pcd.estimate_normals()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_cropped_geometry()

def start_preprocess(config): 
    path = os.path.join(config.input_dir, '*.ply')
    objs = glob.glob(path, recursive=True)
    for obj in objs: 
        cropped = demo_crop_geometry(obj)
        outname = obj[obj.rfind("/")+1:]
        outdir = os.path.join(config.output_dir, outname)
        o3d.io.write_point_cloud(outdir, cropped) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-dir', type=str, default="Segmentation.obj")
    parser.add_argument('-o','--output-dir', type=str, default="test_wpc.ply")
    config = parser.parse_args()
    start_preprocess(config) 
