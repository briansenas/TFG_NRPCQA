# -*- coding: utf-8 -*-

import argparse
import os
import polars as pl 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import scipy
from torch.optim.lr_scheduler import _LRScheduler
from scipy import stats
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import random 
import time
from utils import fit_function

torch.manual_seed(140421)
random.seed(140421)
np.random.seed(140421) 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

from packaging import version        
PYTORCH_VERSION = version.parse(torch.__version__)

class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr

        if num_iter <= 1:
            raise ValueError("`num_iter` must be larger than 1")
        self.num_iter = num_iter

        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        if PYTORCH_VERSION < version.parse("1.1.0"):
            curr_iter = self.last_epoch + 1
            r = curr_iter / (self.num_iter - 1)
        else:
            r = self.last_epoch / (self.num_iter - 1)

        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_all = np.zeros([config.split_num, 4])
    for split in range(config.split_num):
        # model
        if config.pretrained_model_path:
            if not config.model_name == 'Pretrained_mean_with_fast':
                raise ValueError('You must specify the pretrained model path')
            print('The current model is ' + config.model_name)
            model = ResNet_mean_with_fast.resnet50(pretrained=False,feature_fusion_method=config.feature_fusion_method)
            model.load_state_dict(torch.load(config.pretrained_model_path))
            print('Finished Loading the model!')
        else: 
            print('The current model is ' + config.model_name)
            model = ResNet_mean_with_fast.resnet50(pretrained=True,feature_fusion_method=config.feature_fusion_method)
            print('Finished Loading the model!')

        if config.multi_gpu:
            print("[WARNING]: Multi_GPU enable")
            model = torch.nn.DataParallel(model, device_ids=[int(id) for id in config.gpu_ids])
            model = model.to(device)
        else:
            print("Now sending the model to device")
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            size_all_mb = (param_size + buffer_size) / 1024**2
            print('[WARNING]:Model size: {:.3f}MB'.format(size_all_mb))
            model = model.to(device)

        if config.save_predictions: 
            generate_dir('predictions/')

        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))


        print('*************************************************************************************************************')
        print('Using '+ str(split+1) + '-th split.' )

        transformations_train = transforms.Compose([transforms.RandomCrop(224),transforms.ToTensor(),\
                                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        transformations_test = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor(),\
                                                   transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        if config.database == 'SJTU':
            images_dir = 'database/sjtu_2d/'
            datainfo_train = 'database/sjtu_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/sjtu_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = 'database/sjtu_slowfast/'
        elif config.database == 'SJTU_resized':
            images_dir = 'database/sjtu_2d_resized/'
            datainfo_train = 'database/sjtu_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/sjtu_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = 'database/sjtu_slowfast/'
        elif config.database == 'WPC':
            images_dir = 'database/wpc_2d/'
            datainfo_train = 'database/wpc_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/wpc_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = 'database/wpc_slowfast/'
        elif 'LS_SJTU' in config.database:
            images_dir = '../rotation/imgs/'
            datainfo_train = 'database/ls_sjtu_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/ls_sjtu_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = '../extraction/ls_sjtu_features/'
            if 'SCALED' in config.database:
                datainfo_train = 'database/ls_sjtu_data_info_scaled/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/ls_sjtu_data_info_scaled/test_' + str(split+1) +'.csv'
        elif 'OURDATA' in config.database:
            images_dir = '../rotation/ourimgs/'
            datainfo_train = 'database/our_data_info/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/our_data_info/test_' + str(split+1) +'.csv'
            data_3d_dir = '../extraction/our_data_features/'
            if 'SCALED' in config.database:
                datainfo_train = 'database/our_data_info_scaled/train_' + str(split+1) +'.csv'
            datainfo_test = 'database/our_data_info_scaled/test_' + str(split+1) +'.csv'
            if 'RESIZED' in config.database:
                images_dir = '../rotation/ourimgs_resized/'


        trainset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_train, transformations_train, crop_size=config.crop_size,frame_index=config.frame_index,video_length_read = config.video_length_read)
        testset = VideoDataset_NR_image_with_fast_features(images_dir, data_3d_dir, datainfo_test, transformations_test, crop_size=config.crop_size,frame_index=config.frame_index,video_length_read = config.video_length_read)

        ## dataloader
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                                   shuffle=True, num_workers=config.num_workers)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=False, num_workers=config.num_workers)

        # optimizer
        print("Initializing the Optimizer")
        optimizer = optim.Adam(model.parameters(), lr = config.conv_base_lr, weight_decay = 0.0000001)
        criterion = nn.MSELoss().to(device)
        early_stopper = EarlyStopper(patience=6, min_delta=10)
        BATCHES = len(trainset) // config.train_batch_size
        if config.Leslie: 
            num_iters = config.epochs*len(trainset)
            lr_schedule = ExponentialLR(optimizer, 10, num_iters)
            history = {'lr': [], 'loss': []}
            diverge_th = 5 
            smooth_f=0.05
            best_loss = np.inf
            for iteration in range(num_iters):
                i, (video, features, labels, _) = next(enumerate(train_loader))
                video = video.to(device)
                features = features.to(device)
                labels = labels.to(device)
                outputs= model(video, features)
                optimizer.zero_grad()
                loss = criterion(outputs.float(), labels.float())
                loss.backward()
                optimizer.step()
                history['lr'].append(lr_schedule.get_lr()[0])
                lr_schedule.step()
                loss = loss.item()
                if iteration == 0:
                    best_loss = loss
                else:
                    loss = smooth_f * loss + (1 - smooth_f) * history['loss'][-1]
                history['loss'].append(loss)
                if loss > diverge_th * best_loss:
                    print("[LRFINDER] Stopping early, the loss has diverged")
                    break

            min_grad_idx = (np.gradient(np.array(history['loss']))).argmin()
            print(f"[LRFINDER]: best value is {history['lr'][min_grad_idx]}")
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=history['lr'][min_grad_idx], total_steps=num_iters)
        else: 
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio)
        lr = scheduler.get_last_lr()

        best_test_criterion = -1  # SROCC min
        best = np.zeros(4)

        n_test = len(testset)

        print('Starting training:')

        for epoch in range(config.epochs):

            model.train()
            batch_losses = []
            batch_losses_each_disp = []
            for i, (video, features, labels, name) in enumerate(train_loader):
                video = video.to(device)
                features = features.to(device)
                labels = labels.to(device)
                outputs= model(video, features)
                optimizer.zero_grad()
                loss = criterion(outputs.float(), labels.float())
                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())
                loss.backward()
                optimizer.step()
                if config.Leslie: 
                    scheduler.step()


            avg_loss = sum(batch_losses) / (BATCHES)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            if not config.Leslie: 
                scheduler.step()
            lr = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr[0]))


            # Test
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)
            names = []
            # do validation after each epoch
            with torch.no_grad():
                for i, (video, features, labels, name) in enumerate(test_loader):

                    video = video.to(device)
                    features = features.to(device)
                    y_test[i] = labels.item()
                    outputs = model(video, features)
                    y_output[i] = outputs.item()
                    names.append(name[0])

                y_output_logistic = fit_function(y_test, y_output)
                test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
                test_SROCC = stats.spearmanr(y_output, y_test)[0]
                test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
                test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]

                if test_SROCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    if config.save_best_model: 
                        torch.save(model.state_dict(), config.ckpt_path + '/' + config.model_name +'_' + config.database +'_' + str(split+1) + '_' + 'best.pth')
                    if config.save_predictions: 
                        tmp_csv = config.model_name + str(split + 1) + '.csv'
                        predictions = pl.DataFrame(
                            {'predicted_mos': y_output,
                             'normalized_mos': y_output_logistic,
                             'MOS': y_test, 
                             'name': names} 
                        )
                        predictions.write_csv(os.path.join('predictions/', tmp_csv)) 

                    best[0:4] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
                    best_test_criterion = test_SROCC  # update best val SROCC
                    best_all[split, :] = best
                    print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1], best[2], best[3]))
                else:
                    print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1], best[2], best[3]))
                if early_stopper.early_stop(test_SROCC):             
                    break
                print('-------------------------------------------------------------------------------------------------------------------')
        print('Training completed.')
        print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1], best[2], best[3]))
        print('*************************************************************************************************************************')

    performance = np.mean(best_all, 0)
    print('*************************************************************************************************************************')
    print("The mean performance: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(performance[0], performance[1], performance[2], performance[3]))
    print('*************************************************************************************************************************')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=3e-7)
    parser.add_argument('--decay_ratio', type=float, default=0.95)
    parser.add_argument('--decay_interval', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--split_num', type=int, default=9)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--frame_index', type=int, default=5)
    parser.add_argument('--video_length_read', type=int, default=4)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--feature_fusion_method',type=int,default=0)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--losses', type=str, default='./losses')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)

    parser.add_argument('--Leslie', action='store_true', default=False)
    parser.add_argument('--save-best-model', action='store_true', default=False)
    parser.add_argument('--save-predictions', action='store_true', default=False)

    config = parser.parse_args()

    main(config)

