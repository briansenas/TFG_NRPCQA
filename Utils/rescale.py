import os 
import glob 
import os 
from PIL import Image
from torchvision import transforms
import argparse 
from tqdm import tqdm 

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def main(config: dict) -> None: 
    generate_dir(config.output_dir) 
    imgs = glob.glob(os.path.join(config.input_dir, '**/*.png'), recursive=True)
    imgs = sorted(imgs) 
    until = None 
    if __debug__: 
        until = 1

    transform = transforms.Resize(224) 
    for path in tqdm(imgs[:until]): 
        img = Image.open(path)
        # save resized imgs as frames of the video with resolution of (,224)
        img = transform(img)
        img_path = path.split('/') 
        dir = img_path[-2] 
        basename = img_path[-1]
        generate_dir(os.path.join(config.output_dir, dir))
        full_path  = os.path.join(config.output_dir, dir, basename)
        img.save(full_path)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-dir', type=str)
    parser.add_argument('-o', '--output-dir', type=str)
    config = parser.parse_args()
    main(config)

