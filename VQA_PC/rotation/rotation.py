import numpy as np
import time
import open3d as o3d
import os
import math
import numpy as np
import open3d as o3d
import time
from PIL import Image
from torchvision import transforms
import cv2
import argparse
import copy
import multiprocessing
import glob
import functools 
from tqdm import tqdm 

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def camera_rotation(path, img_path, frame_path, video_path, frame_index):
    # We define the image transformation
    transform = transforms.Resize((224, 224)) # fix: pass a tuple instead of an integer
    # Load pointcloud file
    cloud = o3d.io.read_point_cloud(path)

    fps = 15
    size = (1920, 1080)

    video = cv2.VideoWriter(os.path.join(video_path, '030.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    # Generate rotated images
    tmp = 0
    # Define rotation angles
    vid_rate = 30
    rot_deg = 360 / vid_rate
    rot_mat = np.eye(3)
    vis = o3d.visualization.Visualizer() # fix: move outside of the loop
    vis.create_window(visible=False)
    pcd = copy.deepcopy(cloud) # fix: make a copy of the point cloud before rotating
    for tmp in range(1,4*vid_rate+1):
        if tmp < vid_rate:
            rot_mat = np.dot(rot_mat, o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(rot_deg), 0]))
        elif tmp < 2*vid_rate:
            rot_mat = np.dot(rot_mat, o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(rot_deg), 0, 0]))
        elif tmp < 3*vid_rate:
            rot_mat = np.dot(rot_mat, o3d.geometry.get_rotation_matrix_from_axis_angle(
                [np.radians(rot_deg / math.sqrt(2)), np.radians(rot_deg / math.sqrt(2)), 0]))
        elif tmp < 4*vid_rate:
            rot_mat = np.dot(rot_mat, o3d.geometry.get_rotation_matrix_from_axis_angle(
                [np.radians(rot_deg / math.sqrt(2)), -np.radians(rot_deg / math.sqrt(2)), 0]))


        # Rotate pointcloud
        pcd.rotate(rot_mat)

        # Convert pointcloud to image
        vis.clear_geometries()
        vis.add_geometry(pcd) # fix: use the loaded point cloud
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img) * 255).astype(np.uint8))

        # Save the frame_index-th of every 30 frames as the 2D input with resolution of about 1920x1061
        if (tmp - frame_index) % vid_rate == 0:
            img.save(os.path.join(img_path, f"{tmp:03d}.png")) # fix: use f-string formatting instead of str.zfill

        rot_mat = np.eye(3)

        # Save videos
        video.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))

        # Save resized imgs as frames of the video with resolution of (,224)
        img = transform(img)
        img.save(os.path.join(frame_path, f"{tmp:03d}.png"))

    video.release()
    vis.destroy_window() # fix: destroy the visualizer window

def process_object(one_object_path, img_path, frame_path, video_path, frame_index):
    obj = os.path.basename(one_object_path)
    camera_rotation(one_object_path, generate_dir(os.path.join(img_path, obj)), generate_dir(os.path.join(frame_path, obj)), generate_dir(os.path.join(video_path, obj)), frame_index)


def projection(path, img_path, frame_path, video_path, frame_index):
    # find all the .ply objects
    objs = glob.glob(os.path.join(path, '**/*.ply'), recursive=True)
    pool = multiprocessing.Pool()
    pfunc = functools.partial(process_object,img_path=img_path,
                              frame_path=frame_path, video_path=video_path,
                              frame_index=frame_index)
    for result in tqdm(pool.imap(func=pfunc, iterable=objs), total=len(objs)):
        continue 

    # for obj_path in objs:
    #     pool.apply_async(process_object, args=(obj_path, img_path, frame_path, video_path, frame_index))
    # pool.close()
    # pool.join()



def main(config):
    img_path = config.img_path
    frame_path = config.frame_path
    video_path = config.video_path
    generate_dir(img_path)
    generate_dir(frame_path)
    generate_dir(video_path)
    projection(config.path,img_path,frame_path,video_path,config.frame_index)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--path', type=str, default = './ply/') #path to the file that contain .ply models
    parser.add_argument('--img_path', type=str, default = './imgs/') # path to the generated 2D input
    parser.add_argument('--frame_path', type=str, default = './frames/') # path to the generated frames
    parser.add_argument('--video_path', type=str, default = './videos/') # path to the generated videos, disable by default
    parser.add_argument('--frame_index', type=int, default= 5 )
    config = parser.parse_args()

    main(config)
