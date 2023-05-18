import argparse 
import numpy as np
import copy
import open3d as o3d


def demo_crop_geometry(config):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    pcd = o3d.io.read_point_cloud(config.input_dir)
    # o3d.visualization.draw_geometries_with_editing([pcd])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    print(dir(vis)) 
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_cropped_geometry()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_manual_registration(source, config):
    print("Demo for manual ICP")
    target = o3d.io.read_point_cloud(config.output_dir)
    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    assert (len(picked_id_source) >= 3)
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input-dir', type=str, default="Segmentation.obj")
    parser.add_argument('-o','--output-dir', type=str, default="test_wpc.ply")
    config = parser.parse_args()
    cropped = demo_crop_geometry(config)
    o3d.io.write_point_cloud(config.output_dir, cropped) 
