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


def reproject(pixels, depth, device):
    K = get_intrinsics(72/(2*np.pi), depth.shape[1], depth.shape[0], device, return_type='torch')
    pixel_homogeneous = torch.cat((pixels[:,:,0].unsqueeze(dim=2), pixels[:,:,1].unsqueeze(dim=2), torch.ones_like(pixels[:,:,1]).unsqueeze(dim=2)), dim=2)
    pixel_list = (pixel_homogeneous.reshape((-1,3))).type(torch.float64)
    p3d = torch.matmul(torch.inverse(K), pixel_list.T).T * depth.reshape((-1,1)) * 1000
    p3d =(p3d).reshape((depth.shape[0], depth.shape[1], 3))
    return p3d.type(torch.float32)

def get_photometric(flow, img1, img2, pixels2):
    occlusion_mask = get_occlusion_mask(flow, img2, pixels2).unsqueeze(dim=2)
    flow_masked = torch.mul(flow,occlusion_mask)
    pixels1 = (torch.round(pixels2 + flow_masked)).type(torch.long)
    pe = torch.norm(img2 - img1[pixels1[:,:,1], pixels1[:,:,0]], dim=2)
    return pe

def get_geometric(flow, depth1, depth2, pixels2, device):
    occlusion_mask = get_occlusion_mask(flow, depth2, pixels2).unsqueeze(dim=2)
    flow_masked = torch.mul(flow,occlusion_mask)
    pixels1 = torch.round(pixels2 + flow_masked).type(torch.long)
    pixels2_3d = reproject(pixels2, depth2, device)
    pixels1_3d = reproject(pixels1, depth1[pixels1[:,:,1], pixels1[:,:,0]], device)
    p_geo = torch.norm(pixels2_3d - pixels1_3d, dim=2)
    return p_geo

def get_occlusion_mask(flow, img2, pixels2):
    pixels1 = torch.round(pixels2 + flow)
    occlusion_mask = torch.where((pixels1[:,:,0] < img2.shape[1]) * (pixels1[:,:,0] >= 0) * (pixels1[:,:,1] < img2.shape[0]) * (pixels1[:,:,1] >= 0), 1, 0)
    return occlusion_mask

def consensus_loss(ms_scores, img1, img2, depth1, depth2, static_flow, dynamic_flow, pixels, args):
    """
    ms_preds - motion segmentation prediction
    static_pm - static flow photometric error mask
    dynamic_pm - dynamic flow photometric error mask
    photometric error, optical flow error, geometric error
    summing binary cross entropy over all pixels
    """
    consensus_loss = 0

    #Photometric error (static pe_r, dynamic pe_f)
    pe_r = get_photometric(static_flow, img1, img2, pixels.clone())
    pe_f = get_photometric(dynamic_flow, img1, img2, pixels.clone())

    #Optical flow 
    flow_diff = torch.norm(static_flow - dynamic_flow, dim=2)

    #Geometric error (static geo_r, dynamic geo_f)
    geo_r = get_geometric(static_flow, depth1, depth2, pixels.clone(), args.device)
    geo_f = get_geometric(dynamic_flow, depth1, depth2, pixels.clone(), args.device)

    # # Loss Variations
    # label = torch.logical_or(torch.logical_or((pe_r < pe_f), (flow_diff < args.l_C)),(geo_r < geo_f)).unsqueeze(dim=0).type(torch.float)
    label = torch.logical_or((pe_r < pe_f), (flow_diff < args.l_C)).unsqueeze(dim=0).type(torch.float)
    # label = (flow_diff < args.l_C).unsqueeze(dim=0).type(torch.float)

    consensus_loss = nn.functional.binary_cross_entropy(ms_scores, label)
    # consensus_loss = sigmoid_focal_loss(ms_scores, label)

    return consensus_loss

def unsupervised_loss(args, scores, data):
    E_C = 0
    E_M = 0
    sigmoid = nn.Sigmoid()

    ones = torch.ones(1).expand_as(scores).type_as(scores)
    E_M = nn.functional.binary_cross_entropy(sigmoid(scores), ones, reduction='mean')

    for i in range(data[0].shape[0]):
        img1 = data[0][i][:3,:,:].permute(1,2,0)
        img2 = data[0][i][3:,:,:].permute(1,2,0)
        dynamic_flow = data[1][i].permute(1,2,0)
        static_flow = data[2][i].permute(1,2,0)
        depth1 = data[3][i][0].unsqueeze(dim=2)
        depth2 = data[3][i][1].unsqueeze(dim=2)

        pixels = torch.from_numpy(_get_pixel_coords(img2).reshape((512,1382,2))).to(device=args.device)

        E_C += consensus_loss(sigmoid(scores[i]), img1, img2, depth1, depth2, static_flow, dynamic_flow, pixels, args)

    total_loss = args.l_M*E_M + args.l_C*E_C #+ args.l_S*E_S
    return total_loss

def run_val(val_loader, model, epoch, args):
    model.eval()
    val_losses = []
    val_iou = []
    # needed for validation metrics
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for batch_idx, (imgs, dynamic_flow, static_flow, depths, motion_seg) in enumerate(val_loader):
            imgs = imgs.to(args.device).float()
            dynamic_flow = dynamic_flow.to(args.device).float()
            static_flow = static_flow.to(args.device).float()
            depths = depths.to(args.device).float()
            motion_seg = motion_seg.to(args.device)

            # forward
            scores = model(imgs)
            loss = unsupervised_loss(args, scores, (imgs, dynamic_flow, static_flow, depths))

            val_losses.append(loss.item())
            scores_rounded = torch.round(sigmoid(scores))

            if batch_idx == 0:
                args.writer.add_images("visualised_preds", ((scores_rounded.int()*-1)+1), global_step=epoch+1)
                args.writer.add_images("visualised_gts_rgb", imgs[:,3:,:,:], global_step=epoch+1)
                args.writer.add_images("visualised_gts", motion_seg, global_step=epoch+1)

            iou = iou_pytorch(((scores_rounded.int()*-1)+1), motion_seg.int())
            val_iou.append(iou)

        val_loss = sum(val_losses)/len(val_losses)
        IoU = sum(val_iou)/len(val_iou)

        args.writer.add_scalar("val loss", val_loss, epoch*args.train_size)
        args.writer.add_scalar("IoU", IoU, epoch*args.train_size)

    # set back to train ensures layers like dropout, batchnorm are used after eval
    model.train()
    return (val_loss, IoU)

def train(args, train_loader, val_loader, prev_model, logger):
    sigmoid = nn.Sigmoid()

    # init model and pass to `device`
    input_channels=6
    output_channels=1

    if prev_model:
        model = torch.load(prev_model).to(args.device)
        model = model.float()

    else:
        model = UNET(in_channels=input_channels, out_channels=output_channels).to(args.device)
        model = model.float()

    logger.info(f"loaded model of type: {type(model)}")

    # loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor, patience=args.patience, verbose=True)

    # for model saving
    model_name_prefix = args.now.strftime(f"%d-%m-%Y_%H-%M_bs{args.batch_size}")

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
            imgs = imgs.to(args.device).float()
            dynamic_flow = dynamic_flow.to(args.device).float()
            static_flow = static_flow.to(args.device).float()
            depths = depths.to(args.device).float()

            # forward
            scores = model(imgs)
            loss = unsupervised_loss(args, scores, (imgs, dynamic_flow, static_flow, depths))

            # backward
            optimizer.zero_grad()
            loss.backward()

            # adam step
            optimizer.step()

            losses.append(loss.item())

            if batch_idx == 0:
                args.writer.add_images("visualised_preds [training]", ((torch.round(sigmoid(scores))*-1)+1), global_step=epoch+1)

            if (batch_idx) % 20 == 0:
                args.writer.add_scalar("training loss", sum(losses)/len(losses), epoch*steps_per_epoch + batch_idx)
                args.writer.add_scalar("lr change", optimizer.param_groups[0]['lr'], epoch*steps_per_epoch + batch_idx)

        train_loss.append(sum(losses)/len(losses))
        val_IoU.append(run_val(val_loader, model, epoch, args))
        end = time.time()
        total_time += (end-start)
        scheduler.step(val_IoU[epoch][0])

        logger = refresh_logger(args, logger)

        # info logging to log
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] with lr {optimizer.param_groups[0]['lr']}, train loss: {round(train_loss[-1], 5)}, val loss: {round(val_IoU[-1][0], 5)}, IoU: {round(val_IoU[-1][1].item(), 5)}, ETA: {round(((total_time/(epoch+1))*(args.epochs-epoch-1))/60**2,2)} hrs")

        # check if dir exists, if not create one
        models_root = os.path.join(args.root, "saved_models/")
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

    args.writer.close()
    # save final model
    save_path = os.path.join(models_root, model_name_prefix, f"{args.batch_size}_{args.lr}_{args.epochs}.pt")
    torch.save(model, save_path)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-3, type=float, help='Learning rate - default: 5e-5')
    parser.add_argument("--batch_size", default=1, type=int, help='Default=2')
    parser.add_argument("--epochs", default=500, type=int, help='Default=50')
    parser.add_argument("--patience", default=6, type=float, help='Default=3')
    parser.add_argument("--lr_scheduler_factor", default=0.5, type=float, help="Learning rate multiplier - default: 3")
    parser.add_argument("--alpha", default=0.25, type=float, help='Focal loss alpha - default: 0.25')
    parser.add_argument("--gamma", default=2.0, type=float, help='Focal loss gamma - default: 2')
    parser.add_argument("--l_M", default=0.05, type=float, help="hyper-param for motion seg loss")
    parser.add_argument("--l_C", default=0.3, type=float, help="hyper-param for consensus loss")
    parser.add_argument("--l_S", default=1.0, type=float, help="hyper-param for regularization")
    parser.add_argument("--load_chkpt", '-chkpt', default='0', type=str, help="Loading entire checkpoint path for inference/continue training")
    parser.add_argument("--dataset_fraction", default=0.004, type=float, help="fraction of dataset to be used")
    return parser

if __name__ == "__main__":
    args = parse().parse_args()

    args.root = "/storage/remote/atcremers40/motion_seg/"
    # root = "/Carla_Data_Collection/supervised_net"

    data_root = os.path.join(args.root, "datasets/CARLA/")
    log_root = os.path.join(args.root, "logs/")
    root_tb = os.path.join(args.root, "runs/")

    # setup time/date for logging/saving models
    args.now = datetime.now()
    now_string = args.now.strftime(f"%d-%m-%Y_%H-%M_{args.batch_size}_{args.lr}_{args.epochs}")

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
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"running on '{args.device}'")

    # dataset
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CarlaUnsupervised(data_root, test=True) # test kwarg needed for plotting ground truth in tensorboard

    train_loader, val_loader, test_loader = get_dataloaders(dataset, args)

    # initialise tensorboard
    args.writer = SummaryWriter(root_tb + now_string)

    train(args, train_loader, val_loader, prev_model, logger)