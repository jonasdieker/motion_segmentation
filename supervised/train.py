import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.data import DataLoader
from DatasetClass import KITTI_MOD_FIXED, ExtendedKittiMod
from ModelClass import UNET, UNET_Mod
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import PIL
import logging
import argparse

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    
    intersection = torch.sum(torch.bitwise_and(outputs, labels).float(), (1,2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum(torch.bitwise_or(outputs, labels).float(), (1,2))          # Will be zero if both are 0
    
    IoU = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    aIoU = IoU.mean()
        
    return aIoU

def run_val(val_loader, model, epoch, train_size):
    model.eval()
    val_losses = []
    val_iou = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_loader):
            data = data.to(device=device).float()
            targets = targets.to(device=device).float()

            # forward
            scores = model(data)
            loss = sigmoid_focal_loss(scores, targets, alpha=alpha, gamma=gamma, reduction="sum")
            # loss = criterion(scores, y)
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

def train(lr, batch_size, epochs, patience, lr_scheduler_factor, alpha, gamma, prev_model):

    # data split and data loader
    train_size = int(0.8 *  len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

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

    # loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_scheduler_factor, patience=patience, verbose=True)

    # for model saving
    model_name_prefix = now.strftime(f"%d-%m-%Y_%H-%M_bs{batch_size}")

    # train network
    print("train network ...")
    train_loss = []
    val_aIoU = []
    best_val = 1e8
    best_val_epoch = 1
    total_time = 0.0
    for epoch in range(epochs):
        start = time.time()
        model.train()
        losses = []
        steps_per_epoch = len(train_loader)

        for batch_idx, (data, targets) in enumerate(train_loader):

            # move data to gpu if available
            data = data.to(device).float()
            targets = targets.to(device).float()

            # forward
            scores = model(data)
            loss = sigmoid_focal_loss(scores, targets, alpha=alpha, gamma=gamma, reduction="sum")
            # loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # adam step
            optimizer.step()

            losses.append(loss.item())

            if (batch_idx) % 20 == 0:
                writer.add_scalar("training loss", sum(losses)/len(losses), epoch*steps_per_epoch + batch_idx)
                writer.add_scalar("lr change", optimizer.param_groups[0]['lr'], epoch*steps_per_epoch + batch_idx)

        # print(f"Epoch {epoch}: loss => {sum(losses)/len(losses)}")
        train_loss.append(sum(losses)/len(losses))
        val_aIoU.append(run_val(val_loader, model, epoch, train_size))
        end = time.time()
        total_time += (end-start)
        scheduler.step(val_aIoU[epoch][0])
        logger.info(f"Epoch [{epoch + 1}/{epochs}] with lr {optimizer.param_groups[0]['lr']}, train loss: {round(train_loss[-1], 5)}, val loss: {round(val_aIoU[-1][0], 5)}, aIoU: {round(val_aIoU[-1][1].item(), 5)}, ETA: {round(((total_time/(epoch+1))*(epochs-epoch-1))/60**2,2)} hrs")

        # Logging fix for stale file handler
        logger.removeHandler(logger.handlers[1])
        fh = logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # check if dir exists, if not create one
        models_root = f"/storage/remote/atcremers40/motion_seg/saved_models/"
        if not os.path.isdir(os.path.join(models_root, model_name_prefix)):
            os.mkdir(os.path.join(models_root, model_name_prefix), mode=0o770)
        if (epoch+1) % 5 == 0:
            # save interim model
            save_path = os.path.join(models_root, f"{model_name_prefix}/{batch_size}_{lr}_{epoch+1}.pt")
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
            logger.info(f"Epoch [{epoch + 1}] Current best learning rate at epoch {best_aIoU_epoch}")

    writer.close()
    # save final model
    save_path = os.path.join(models_root, model_name_prefix, f"{batch_size}_{lr}_{epochs}.pt")
    torch.save(model, save_path)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-3, type=int, help='Learning rate - default: 5e-3')
    parser.add_argument("--batch_size", default=2, type=int, help='Default=2')
    parser.add_argument("--epochs", default=75, type=int, help='Default=75')
    parser.add_argument("--patience", default=4, type=float, help='Default=4')
    parser.add_argument("--lr_scheduler_factor", default=0.25, type=float, help="Learning rate multiplier - default: 3")
    parser.add_argument("--alpha", default=0.25, type=float, help='Focal loss alpha - default: 0.25')
    parser.add_argument("--gamma", default=2.0, type=float, help='Focal loss gamma - default: 2')
    parser.add_argument("--load_chkpt", '-chkpt', default='0', type=str, help="Loading entire checkpoint path for inference/continue training")
    return parser

if __name__ == "__main__":
    args = parse().parse_args()

    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    lr_scheduler_factor = args.lr_scheduler_factor
    alpha = args.alpha
    gamma = args.gamma

    # load checkpoint if path exists
    if os.path.exists(args.load_chkpt):
        prev_model = args.load_chkpt
        print(f"Loading model {os.path.basename(prev_model)} from {os.path.dirname(prev_model)}")
    elif args.load_chkpt == '0':
        prev_model=None
    else:
        prev_model=None
        print("Path specified incorrectly, training without a checkpoint model")

    # specify some hyperparams
    print(f"running with lr={lr}, batch_size={batch_size}, epochs={epochs}")

    # setup time/date for logging/saving models
    now = datetime.now()
    now_string = now.strftime(f"%d-%m-%Y_%H-%M_{batch_size}_{lr}_{epochs}")

    # setup logging
    log_root = "/storage/remote/atcremers40/motion_seg/logs"
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
    ])
    logger = logging.getLogger()
    
    # set device and clean up
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    print(f"running on '{device}'")

    # dataset
    # data_root = '/storage/remote/atcremers40/motion_seg/datasets/KITTI_MOD_fixed/training/'
    data_root = "/storage/remote/atcremers40/motion_seg/datasets/Extended_MOD_Masks/"
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    # dataset = KITTI_MOD_FIXED(data_root, data_transforms)
    dataset = ExtendedKittiMod(data_root)

    # initialise tensorboard
    writer = SummaryWriter("/storage/remote/atcremers40/motion_seg/runs/" + now_string)

    # needed for validation metrics
    sigmoid = nn.Sigmoid()

    train(lr, batch_size, epochs, patience, lr_scheduler_factor, alpha, gamma, prev_model)