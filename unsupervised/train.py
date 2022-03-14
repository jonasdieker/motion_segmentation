import sys
import os
import gc
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.ops.focal_loss import sigmoid_focal_loss
from DatasetClass import CarlaUnsupervised
from ModelClass import UNET, UNET_Mod
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from numpy.matlib import repmat
# import matplotlib.pyplot as plt
import PIL
import logging
import argparse

from utils_data import get_flow, reproject, get_intrinsics
from utils_train import setup_logger, refresh_logger, get_dataloaders, iou_pytorch, _get_pixel_coords


def get_photometric_error(flow, coords2, img1, img2):
    pixel2 = img2[coords2[0], coords2[1]]
    coords1 = coords2 + flow[coords2[0], coords2[1]].clone().detach().cpu().numpy()
    if coords1[0] < img1.shape[0] and coords1[0] >= 0 and coords1[1] < img1.shape[1] and coords1[1] >=0:
        pixel1 = img1[coords1[0], coords1[1]]
    else:
        pixel1 = pixel2 # value not known

    p_e = np.sum(np.linalg.norm(pixel2 - pixel1))
    return p_e

def get_geometric_error(flow, depth1, depth2, coords2):
    image_size_x = depth2.shape[1]
    image_size_y = depth2.shape[0]
    pixel2_3d = reproject_pixel(coords2, depth2[coords2[0], coords2[1]], image_size_x, image_size_y)
    coords1 = coords2 + flow[coords2[0], coords2[1]]
    if coords1[0] < img1.shape[0] and coords1[0] >= 0 and coords1[1] < img1.shape[1] and coords1[1] >=0:
        pixel1_3d = reproject_pixel(coords1, depth1[coords1[0], coords1[1]], image_size_x, image_size_y)
    else:
        pixel1_3d = pixel2_3d

    p_geo = np.sum(np.linalg.norm(pixel2_3d - pixel1_3d))
    return p_geo
    
def reproject_pixel(pixel, depth, image_size_x, image_size_y):
    K = get_intrinsics(72/(2*np.pi), image_size_x, image_size_y)
    pixel_homogeneous = np.array([pixel[0], pixel[1], 1])
    p3d = np.dot(np.linalg.inv(K), pixel_homogeneous) * depth * 1000
    return p3d

def consensus_loss(ms_scores, img1, img2, depth1, depth2, static_flow, dynamic_flow, l_C):
    """
    ms_preds - motion segmentation prediction
    static_pm - static flow photometric error mask
    dynamic_pm - dynamic flow photometric error mask
    photometric error, optical flow error, geometric error
    summing binary cross entropy over all pixels
    """
    consensus_loss = 0

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            #Photometric error (static pe_r, dynamic pe_f)
            pe_r = get_photometric_error(static_flow, np.array([i,j]), img1, img2)
            pe_f = get_photometric_error(dynamic_flow, np.array([i,j]), img1, img2)

            #Optical flow 
            flow_diff = np.sum(np.linalg.norm(static_flow[i,j] - dynamic_flow[i,j]))

            #geometric error (static geo_r, dynamic geo_f)
            geo_r = get_geometric_error(static_flow, depth1, depth2, np.array(i,j))
            geo_f = get_geometric_error(dynamic_flow, depth1, depth2, np.array(i,j))
        
            label = (pe_r < pe_f) | (flow_diff > l_C) | (geo_r < geo_f)
            consensus_loss += nn.functional.binary_cross_entropy(ms_scores, label)

    return consensus_loss

def unsupervised_loss(scores, data, l_M = 1, l_C = 1, l_S = 1):
    E_M = 0
    E_C = 0

    for i in range(data[0].shape[0]):
        img1 = data[0][i][:3,:,:].permute(1,2,0)
        img2 = data[0][i][3:,:,:].permute(1,2,0)
        dynamic_flow = data[1][i].permute(1,2,0)
        static_flow = data[2][i].permute(1,2,0)
        depth1 = data[3][i][0].unsqueeze(dim=2)
        depth2 = data[3][i][1].unsqueeze(dim=2)

        ones = torch.ones(1).expand_as(scores).type_as(scores)
        E_M += nn.functional.binary_cross_entropy(scores, ones, reduction='sum')
        E_C += consensus_loss(scores, img1, img2, depth1, depth2, static_flow, dynamic_flow, l_C)

    total_loss = l_M*E_M + l_C*E_C #+ l_S*E_S
    return total_loss

def run_val(val_loader, model, epoch, args):
    model.eval()
    val_losses = []
    val_iou = []
    # needed for validation metrics
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data.to(device=device).float()
            motion_seg = data[:,4]

            # forward
            scores = model(data[:,0])
            loss = unsupervised_loss(scores, data)

            val_losses.append(loss.item())
            scores_rounded = torch.round(sigmoid(scores))

            if batch_idx == 0:
                writer.add_images("visualised_preds", scores_rounded, global_step=epoch+1)
                writer.add_images("visualised_gts_rgb", data[:,0,3:6,:,:], global_step=epoch+1)
                writer.add_images("visualised_gts", motion_seg, global_step=epoch+1)

            iou = iou_pytorch(scores_rounded.int(), motion_seg.int())
            val_iou.append(iou)

        val_loss = sum(val_losses)/len(val_losses)
        IoU = sum(val_iou)/len(val_iou)

        writer.add_scalar("val loss", val_loss, epoch*args.train_size)
        writer.add_scalar("IoU", IoU, epoch*args.train_size)

    # set back to train ensures layers like dropout, batchnorm are used after eval
    model.train()
    return (val_loss, IoU)

def train(args, prev_model, logger):

    # init model and pass to `device`
    input_channels=6
    output_channels=1

    if prev_model:
        model = torch.load(prev_model).to(device)
        model = model.float()

    else:
        model = UNET(in_channels=input_channels, out_channels=output_channels).to(device)
        # model = UNET_Mod(input_channels, output_channels).to(device)
        model = model.float()

    logger.info(f"loaded model of type: {type(model)}")

    # loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor, patience=args.patience, verbose=True)

    # for model saving
    model_name_prefix = now.strftime(f"%d-%m-%Y_%H-%M_bs{args.batch_size}")

    # train network
    print("train network ...")
    train_loss = []
    val_IoU = []
    best_val = 0
    best_val_epoch = 1
    total_time = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        losses = []
        steps_per_epoch = len(train_loader)

        for batch_idx, (imgs, dynamic_flow, static_flow, depths, _) in enumerate(train_loader):
            # move data to gpu if available
            imgs = imgs.to(device).float()
            dynamic_flow = dynamic_flow.to(device).float()
            static_flow = static_flow.to(device).float()
            depths = depths.to(device).float()

            # forward
            scores = model(imgs)
            loss = unsupervised_loss(scores, (imgs, dynamic_flow, static_flow, depths))

            # backward
            optimizer.zero_grad()
            loss.backward()

            # adam step
            optimizer.step()

            losses.append(loss.item()) 

            if (batch_idx) % 20 == 0:
                writer.add_scalar("training loss", sum(losses)/len(losses), epoch*steps_per_epoch + batch_idx)
                writer.add_scalar("lr change", optimizer.param_groups[0]['lr'], epoch*steps_per_epoch + batch_idx)

        train_loss.append(sum(losses)/len(losses))
        val_IoU.append(run_val(val_loader, model, epoch, args))
        end = time.time()
        total_time += (end-start)
        scheduler.step(val_IoU[epoch][0])

        logger = refresh_logger(args, logger)

        # info logging to log
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] with lr {optimizer.param_groups[0]['lr']}, train loss: {round(train_loss[-1], 5)}, val loss: {round(val_IoU[-1][0], 5)}, IoU: {round(val_IoU[-1][1].item(), 5)}, ETA: {round(((total_time/(epoch+1))*(args.epochs-epoch-1))/60**2,2)} hrs")

        # check if dir exists, if not create one
        models_root = f"/storage/remote/atcremers40/motion_seg/saved_models/"
        if not os.path.isdir(os.path.join(models_root, model_name_prefix)):
            os.mkdir(os.path.join(models_root, model_name_prefix), mode=0o770)
        if (epoch+1) % 5 == 0:
            # save interim model
            save_path = os.path.join(models_root, f"{model_name_prefix}/{args.batch_size}_{args.lr}_{epoch+1}.pt")
            torch.save(model, save_path)

        #Saving the best IoU model
        if val_IoU[-1][1].item() >= best_val:
            best_val = val_IoU[-1][1].item()
            best_IoU_epoch = epoch+1
            save_path = os.path.join(models_root, f"{model_name_prefix}/best_IoU.pt")
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(model, save_path)
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1} Current best IoU at epoch {best_IoU_epoch}")

    writer.close()
    # save final model
    save_path = os.path.join(models_root, model_name_prefix, f"{args.batch_size}_{args.lr}_{args.epochs}.pt")
    torch.save(model, save_path)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1.25e-5, type=float, help='Learning rate - default: 5e-3')
    parser.add_argument("--batch_size", default=1, type=int, help='Default=2')
    parser.add_argument("--epochs", default=50, type=int, help='Default=50')
    parser.add_argument("--patience", default=6, type=float, help='Default=3')
    parser.add_argument("--lr_scheduler_factor", default=0.5, type=float, help="Learning rate multiplier - default: 3")
    parser.add_argument("--alpha", default=0.25, type=float, help='Focal loss alpha - default: 0.25')
    parser.add_argument("--gamma", default=2.0, type=float, help='Focal loss gamma - default: 2')
    parser.add_argument("--l_M", default=1.0, type=float, help="hyper-param for motion seg loss")
    parser.add_argument("--l_C", default=1.0, type=float, help="hyper-param for consensus loss")
    parser.add_argument("--l_S", default=1.0, type=float, help="hyper-param for regularization")
    parser.add_argument("--load_chkpt", '-chkpt', default='0', type=str, help="Loading entire checkpoint path for inference/continue training")
    parser.add_argument("--dataset_fraction", default=1.0, type=float, help="fraction of dataset to be used")
    return parser

if __name__ == "__main__":
    args = parse().parse_args()
    root_tb = "/storage/remote/atcremers40/motion_seg/runs/"
    data_root = "/storage/remote/atcremers40/motion_seg/datasets/Opt_flow_pixel_preprocess"
    log_root = "/storage/remote/atcremers40/motion_seg/logs"

    # setup time/date for logging/saving models
    now = datetime.now()
    now_string = now.strftime(f"%d-%m-%Y_%H-%M_{args.batch_size}_{args.lr}_{args.epochs}")

    # setup logging
    args, logger = setup_logger(args, log_root, now_string)

    # load checkpoint if path exists
    if os.path.exists(args.load_chkpt):
        prev_model = args.load_chkpt
        logger.info(f"Loading model {os.path.basename(prev_model)} from {os.path.dirname(prev_model)}")
    elif args.load_chkpt == '0':
        prev_model=None
    else:
        prev_model=None
        logger.warning("Path specified incorrectly, training without a checkpoint model")

    # specify some hyperparams
    logger.info(f"running with lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, patience={args.patience}, lr_scheduler_factor={args.lr_scheduler_factor} alpha={args.alpha}, gamma={args.gamma}")
    
    # set device and clean up
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"running on '{device}'")

    # dataset
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CarlaUnsupervised(data_root, test=True) # test kwarg needed for plotting ground truth in tensorboard

    train_loader, val_loader, test_loader = get_dataloaders(dataset, args)

    # initialise tensorboard
    writer = SummaryWriter(root_tb + now_string)

    train(args, prev_model, logger)