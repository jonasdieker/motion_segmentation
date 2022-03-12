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

from utils_data import get_flow, reproject
from utils_train import setup_logger, refresh_logger, get_dataloaders, iou_pytorch, _get_pixel_coords


def get_photometric_error(flow, img1, img2):

    coords1 = _get_pixel_coords(img1)
    image_size_x = img1.shape[1]
    image_size_y = img1.shape[0]

    # reshape images
    I1 = img1.reshape((-1,3))
    I2 = img2.reshape((-1,3))

    flow_reshaped = flow.reshape((-1,2))
    coords2 = (coords1 + flow_reshaped).astype(int)
    in_image = np.invert((coords2[:,0] < 0) | (coords2[:,0] > image_size_x) | (coords2[:,1] < 0) | (coords2[:,1] > image_size_y))
    I1_masked = I1[in_image, :]
    coords2_masked = coords2[in_image, :]
    I2 = I2[coords2_masked[:,1]*image_size_y+coords2_masked[:,0], :]

    return np.linalg.norm(I1_masked-I2, axis = 1)


def get_geometric_error(static_flow, img1, img2, l_geo = 5):

    geometric_mask = 0

    return geometric_mask
    
    # get pixel correspondences using flow
    # check if still in image 2
    # project pixels in both images to 3D
    # take l2 norm between 3D points
    # define some threshold lambda_geo
    # return sum


def consensus_loss(ms_scores, img1, img2, static_flow, dynamic_flow):
    """
    ms_preds - motion segmentation prediction

    static_pm - static flow photometric error mask
    dynamic_pm - dynamic flow photometric error mask

    photometric error, optical flow error, geometric error

    summing binary cross entropy over all pixels

    """
    # reshape images
    I1 = img1.reshape((-1,3))
    I2 = img2.reshape((-1,3))
    static_flow_reshaped = static_flow.reshape((-1,2))
    dynamic_flow_reshaped = dynamic_flow.reshape((-1,2))

    ms_preds = ms_scores.reshape((-1,2))

    pixels = _get_pixel_coords(img1)

    for i in range(len(pixels)):
        #photometric error (static pe_r, dynamic pe_f)
        pe_r = 0
        pe_f = 0

        #Optical flow 

        u = dynamic_flow[i]

        #geometric error (static geo_r, dynamic geo_f)
        geo_r = 0
        geo_f = 0
        
        #opt
        # gt_indicator = (pe>0) | opt_flow_e >0 | geo_e > 0
    #     pixel_loss = nn.functional.binary_cross_entropy(total_mask, ms_scores)


    # static_pe_mask = get_photometric_error(static_flow, img1, img2)

    # dynamic_pe_mask = get_photometric_error(dynamic_flow, img1, img2)
    # pe_mask = static_pe_mask < dynamic_pe_mask

    # opt_flow_mask = get_opt_flow_mask(static_flow, dynamic_flow)

    # geometric_mask = get_geometric_error(static_flow, img1, img2)

    # total_mask = pe_mask | opt_flow_mask | geometric_mask

    # consensus_loss = nn.functional.binary_cross_entropy(total_mask, ms_scores)

    # return consensus_loss
    return 0


def unsupervised_loss(scores, data, l_M = 1, l_E = 1, l_S = 1):
    # E = lambda_M * E_M + lambda_C * E_C + lambda_S * E_S

    dynamic_flow = data[1] #512x1382x2
    static_flow = data[2] #512x1382x2
    img1 = data[0][:3,:,:].permute(1,2,0) #3x512x1382 -> 512x1382x3
    img2 = data[0][3:,:,:].permute(1,2,0)

    ones = torch.ones(1).expand_as(scores).type_as(scores)
    E_M = nn.functional.binary_cross_entropy(scores, ones)

    E_C = consensus_loss(scores, img1, img2, static_flow, dynamic_flow)

    total_loss = l_M*E_M + l_M*E_C #+ l_S*E_S

    return total_loss


def run_val(val_loader, model, epoch, train_size):
    model.eval()
    val_losses = []
    val_iou = []
    # needed for validation metrics
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(device=device).float()
            targets = targets.to(device=device).float()

            # forward
            scores = model(data)
            # loss = sigmoid_focal_loss(scores, targets, alpha=alpha, gamma=gamma, reduction="sum")
            # loss = criterion(scores, y)
            loss = unsupervised_loss(scores, data)

            val_losses.append(loss.item())
            scores_rounded = torch.round(sigmoid(scores))

            if batch_idx == 0:
                writer.add_images("visualised_preds", scores_rounded, global_step=epoch+1)
                writer.add_images("visualised_gts_rgb", data[:,0:3,:,:], global_step=epoch+1)
                writer.add_images("visualised_gts", targets, global_step=epoch+1)

            iou = iou_pytorch(scores_rounded.int(), targets.int())
            val_iou.append(iou)

        val_loss = sum(val_losses)/len(val_losses)
        aIoU = sum(val_iou)/len(val_iou)

        writer.add_scalar("val loss", val_loss, epoch*train_size)
        writer.add_scalar("aIoU", aIoU, epoch*train_size)

    # set back to train ensures layers like dropout, batchnorm are used after eval
    model.train()
    return (val_loss, aIoU)

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
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor, patience=args.patience, verbose=True)

    # for model saving
    model_name_prefix = now.strftime(f"%d-%m-%Y_%H-%M_bs{args.batch_size}")

    # train network
    print("train network ...")
    train_loss = []
    val_aIoU = []
    best_val = 1e8
    best_val_epoch = 1
    total_time = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        losses = []
        steps_per_epoch = len(train_loader)

        for batch_idx, (imgs, dynamic_flow, static_flow, depths) in enumerate(train_loader):

            # move data to gpu if available
            imgs = imgs.to(device).float()
            dynamic_flow = dynamic_flow.to(device).float()
            static_flow = static_flow.to(device).float()
            depths = depths.to(device).float()

            # forward
            scores = model(imgs)
            # loss = sigmoid_focal_loss(scores, targets, alpha=alpha, gamma=gamma, reduction="sum")
            # loss = criterion(scores, targets)
            loss = unsupervised_loss(scores, imgs)

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
        val_aIoU.append(run_val(val_loader, model, epoch, args.train_size))
        end = time.time()
        total_time += (end-start)
        scheduler.step(val_aIoU[epoch][0])

        logger = refresh_logger(args, logger)

        # info logging to log
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] with lr {optimizer.param_groups[0]['lr']}, train loss: {round(train_loss[-1], 5)}, val loss: {round(val_aIoU[-1][0], 5)}, aIoU: {round(val_aIoU[-1][1].item(), 5)}, ETA: {round(((total_time/(epoch+1))*(args.epochs-epoch-1))/60**2,2)} hrs")

        # check if dir exists, if not create one
        models_root = f"/storage/remote/atcremers40/motion_seg/saved_models/"
        if not os.path.isdir(os.path.join(models_root, model_name_prefix)):
            os.mkdir(os.path.join(models_root, model_name_prefix), mode=0o770)
        if (epoch+1) % 5 == 0:
            # save interim model
            save_path = os.path.join(models_root, f"{model_name_prefix}/{args.batch_size}_{args.lr}_{epoch+1}.pt")
            torch.save(model, save_path)

        #Saving the best aIoU model
        if val_aIoU[-1][1] < best_val:
            best_aIoU = val_aIoU[-1][1]
            best_aIoU_epoch = epoch+1
            save_path = os.path.join(models_root, f"{model_name_prefix}/best_aIoU.pt")
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(model, save_path)
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1}] Current best aIoU at epoch {best_aIoU_epoch}")

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
    parser.add_argument("--load_chkpt", '-chkpt', default='0', type=str, help="Loading entire checkpoint path for inference/continue training")
    parser.add_argument("--dataset_fraction", default=1.0, type=float, help="fraction of dataset to be used")
    return parser

if __name__ == "__main__":
    args = parse().parse_args()
    root_tb = "/storage/remote/atcremers40/motion_seg/runs/"
    data_root = "/storage/remote/atcremers40/motion_seg/datasets/"
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
    dataset = CarlaUnsupervised(data_root)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, args.batch_size, args.dataset_fraction)

    # initialise tensorboard
    writer = SummaryWriter(root_tb + now_string)

    train(args, prev_model, logger)