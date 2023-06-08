# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
from torchvision import transforms
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import pandas as pd
import random
from utils import fit_function

torch.manual_seed(140421)
random.seed(140421)
np.random.seed(140421) 


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model
    model = ResNet_mean_with_fast.resnet50(pretrained=False, feature_fusion_method=config.feature_fusion_method)
    model.load_state_dict(torch.load(config.pretrained_model_path))
    model = model.to(device)
    model.eval()
    ## training data
    images_dir = config.path_imgs
    data_3d_dir = config.path_3d_features
    datainfo_test = config.data_info
    transformations_test = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, transformations_test, crop_size=224)
    
    ## initialize dataloader
    n_test = len(testset)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
    y_output = np.zeros(n_test)
    y_test = np.zeros(n_test)
    y_name = ['video_name'] * n_test

    # begin inference
    with torch.no_grad():
        for i, (imgs, features, labels, video_name) in enumerate(test_loader):  
            # print(video_name[0])
            imgs = imgs.to(device)
            features = features.to(device)
            y_test[i] = labels.item()
            outputs = model(imgs, features)
            y_output[i] = outputs.item()
            y_name[i] =  video_name[0]
        y_output_logistic = fit_function(y_test, y_output)
        test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
        test_SROCC = stats.spearmanr(y_output, y_test)[0]
        test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
        test_KROCC = stats.kendalltau(y_output, y_test)[0]
        print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))

    data = pd.DataFrame({'vid_name':y_name,'predicted_mos':y_output_logistic})
    data.to_csv(config.output_csv_path, index = None)

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--path_imgs', type=str)
    parser.add_argument('--path_3d_features', type=str)
    parser.add_argument('--data_info', type=str)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--output_csv_path', type=str, default = 'prediction.csv')
    parser.add_argument('--feature_fusion_method', type=str, default = 2)

    config = parser.parse_args()

    main(config)


