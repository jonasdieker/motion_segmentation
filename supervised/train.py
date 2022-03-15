import os
import gc
from datetime import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DatasetClass import KITTI_MOD_FIXED, ExtendedKittiMod, CarlaMotionSeg
from ModelClass import UNET, UNET_Mod
from utils_train import setup_logger, refresh_logger, get_dataloaders, iou_pytorch

def run_val(val_loader, model, epoch, args):
    model.eval()
    val_losses = []
    val_iou = []
    # needed for validation metrics
    sigmoid = nn.Sigmoid()
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(device=args.device).float()
            targets = targets.to(device=args.device).float()

            # forward
            scores = model(data)
            if args.loss_type == 'focal':
                loss = sigmoid_focal_loss(scores, targets, alpha=args.alpha, gamma=args.gamma, reduction="sum")
            else:
                loss = criterion(scores, targets)

            val_losses.append(loss.item())
            scores_rounded = torch.round(sigmoid(scores))

            if batch_idx == 0:
                args.writer.add_images("visualised_preds", scores_rounded, global_step=epoch+1)
                args.writer.add_images("visualised_gts_rgb", data[:,0:3,:,:], global_step=epoch+1)
                args.writer.add_images("visualised_gts", targets, global_step=epoch+1)

            iou = iou_pytorch(scores_rounded.int(), targets.int())
            val_iou.append(iou)

        val_loss = sum(val_losses)/len(val_losses)
        IoU = sum(val_iou)/len(val_iou)

        args.writer.add_scalar("val loss", val_loss, epoch*args.train_size/args.batch_size)
        args.writer.add_scalar("IoU", IoU, epoch*args.train_size/args.batch_size)

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
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor, patience=args.patience, verbose=True)

    # for model saving
    model_name_prefix = args.now.strftime(f"%d-%m-%Y_%H-%M_bs{args.batch_size}")

    # train network
    print("train network ...")
    train_loss = []
    val_IoU = []
    best_val = 0
    best_IoU_epoch = 1
    total_time = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        losses = []
        steps_per_epoch = len(train_loader)

        for batch_idx, (data, targets) in enumerate(train_loader):

            # move data to gpu if available
            data = data.to(args.device).float()
            targets = targets.to(args.device).float()

            # forward
            scores = model(data)
            if args.loss_type == 'focal':
                loss = sigmoid_focal_loss(scores, targets, alpha=args.alpha, gamma=args.gamma, reduction="sum")
            else:
                loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # adam step
            optimizer.step()

            losses.append(loss.item()) 

            if batch_idx == 0:
                args.writer.add_images("visualised_preds [training]", torch.round(sigmoid(scores)), global_step=epoch+1)

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
    parser.add_argument("--lr", default=1.25e-5, type=float, help='Learning rate - default: 5e-3')
    parser.add_argument("--batch_size", default=2, type=int, help='Default=2')
    parser.add_argument("--epochs", default=50, type=int, help='Default=50')
    parser.add_argument("--loss_type", default='focal', type=str, help='Loss types available - focal, bce')
    parser.add_argument("--patience", default=3, type=int, help='Default=3')
    parser.add_argument("--lr_scheduler_factor", default=0.5, type=float, help="Learning rate multiplier - default: 3")
    parser.add_argument("--alpha", default=0.25, type=float, help='Focal loss alpha - default: 0.25')
    parser.add_argument("--gamma", default=2.0, type=float, help='Focal loss gamma - default: 2')
    parser.add_argument("--load_chkpt", '-chkpt', default='0', type=str, help="Loading entire checkpoint path for inference/continue training")
    parser.add_argument("--dataset_fraction", default=0.02, type=float, help="fraction of dataset to be used")
    return parser

if __name__ == "__main__":
    args = parse().parse_args()

    root = "/storage/remote/atcremers40/motion_seg/"
    # root = "/Carla_Data_Collection/supervised_net"

    # data_root = os.path.join(root, "datasets/KITTI_MOD_fixed/training/")
    # data_root = os.path.join(root, "datasets/Extended_MOD_Masks/")
    # data_root = os.path.join(root, "datasets/Carla_Annotation/Carla_Export/")
    data_root = os.path.join(root, "datasets/Opt_flow_pixel_preprocess/")
    log_root = os.path.join(root, "logs/")
    root_tb = os.path.join(root, "runs/")

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

    logger.info(f"running with lr={args.lr}, batch_size={args.batch_size}, epochs={args.epochs}, loss_type = {args.loss_type}, patience={args.patience}, lr_scheduler_factor={args.lr_scheduler_factor} alpha={args.alpha}, gamma={args.gamma}")
    
    # set device and clean up
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"running on '{args.device}'")

    # dataset = ExtendedKittiMod(data_root)
    dataset = CarlaMotionSeg(data_root)

    train_loader, val_loader, test_loader = get_dataloaders(dataset, args)

    # initialize tensorboard
    args.writer = SummaryWriter(os.path.join(root_tb, now_string))

    train(args, train_loader, val_loader, prev_model, logger)