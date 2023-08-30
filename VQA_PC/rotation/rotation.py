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

# Camera Rotation
def camera_rotation(path, img_path,frame_path,video_path,frame_index):
    pcd = o3d.io.read_point_cloud(path)
    transform = transforms.Resize(224) 
    if not os.path.exists(img_path+'/'):
        os.mkdir(img_path+'/')
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ctrl = vis.get_view_control()
    tmp = 0
    interval = 5.82
    #fps = 30 
    #size = (405, 224) 
    #video = cv2.VideoWriter(video_path + '/' + 'video.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    # interval represent rotation interval per degree
    # 12 * interval indicates 12 degreee
    start = time.time()

    # begin rotation
    while tmp<120:
        tmp+=1
        if tmp<30:
            ctrl.rotate(12*interval, 0)
        elif tmp>=30 and tmp<60:
            ctrl.rotate(0, 12*interval)
        elif tmp>=60 and tmp<90:
            ctrl.rotate(12*interval/math.sqrt(2), 12*interval/math.sqrt(2))
        elif tmp>=90 and tmp<120:
            ctrl.rotate(12*interval/math.sqrt(2), -12*interval/math.sqrt(2))
        vis.poll_events()
        vis.update_renderer()    
        img = vis.capture_screen_float_buffer(True)
        img = Image.fromarray((np.asarray(img)* 255).astype(np.uint8))
        # save resized imgs as frames of the video with resolution of (,224)
        img = transform(img)

        # save the fram_index -th of every 30 frames as the 2D input with resolution of about 1920x1061
        if (tmp-frame_index) % 30 == 0:
            img.save(img_path + '/'+str(tmp).zfill(3)+'.png')

        img.save(frame_path + '/'+str(tmp).zfill(3)+'.png')

        # save videos
        #video.write(cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR))

    end = time.time()
    vis.destroy_window()
    del ctrl
    del vis

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
