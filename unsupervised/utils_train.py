import sys
import os
import gc
from datetime import datetime
import time
import logging
import numpy as np
from numpy.matlib import repmat

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.data import DataLoader


#-------------------------- LOGGING ---------------------------#

def setup_logger(args, log_root, now_string):
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_root, f'{now_string}.log'))
    ])
    logger = logging.getLogger()

    args.log_root = log_root
    args.formatter = formatter
    args.now_string = now_string
    return args, logger

def refresh_logger(args, logger):
    # Logging fix for stale file handler
    logger.removeHandler(logger.handlers[1])
    fh = logging.FileHandler(os.path.join(args.log_root, f'{args.now_string}.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(args.formatter)
    logger.addHandler(fh)
    return logger

#----------------------------- DATA LOADING ----------------------#

def get_dataloaders(dataset, args, shuffle=True, overfit=False):
    dataset_length = int(len(dataset) * args.dataset_fraction)
    train_size = int(0.6 *  dataset_length)
    args.train_size = train_size
    val_size = int(0.5*(dataset_length - train_size))
    test_size = dataset_length - train_size - val_size
    if overfit:
        train_size = 1
        val_size = dataset_length - train_size -1
        test_size = 1
        shuffle = False
    # taking subset of dataset according to dataset_fraction
    dataset_idx = list(range(0, dataset_length))
    dataset = torch.utils.data.Subset(dataset, dataset_idx)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader

def load_model(model_path, device='cuda:0'):
    if device == "cuda:0":
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            device = torch.device(device)
            print("Moving model to GPU")
            return torch.load(model_path).to(device)
        else:
            print("No GPU available!")

    if device == "cpu":
        print("Model on CPU")
        return torch.load(model_path, map_location=lambda storage, loc: storage)

#-------------------------------- VALIDATION -------------------------#

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    
    intersection = torch.sum(torch.bitwise_and(outputs, labels).float(), (1,2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum(torch.bitwise_or(outputs, labels).float(), (1,2))          # Will be zero if both are 0
    
    IoU = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    aIoU = IoU.mean()
        
    return aIoU

#------------------------------------ LOSSES ----------------------------#

def _get_pixel_coords(img1):
    # 2d pixel coordinates
    image_size_x = img1.shape[1]
    image_size_y = img1.shape[0]
    pixel_length = image_size_x * image_size_y
    u_coords = repmat(np.r_[image_size_x-1:-1:-1],
                        image_size_y, 1).reshape(pixel_length)
    v_coords = repmat(np.c_[image_size_y-1:-1:-1],
                        1, image_size_x).reshape(pixel_length)

    u_coords = np.flip(u_coords).reshape(-1)
    v_coords = np.flip(v_coords).reshape(-1)
    coords1 = np.stack((u_coords, v_coords), axis=-1)
    return coords1

def get_opt_flow_mask(static_flow, dynamic_flow, l_C = 10):
    """
    static_flow.shape: (512, 1382, 2)
    """
    combo_mask = np.linalg.norm(np.sum((dynamic_flow, -static_flow), axis=0), axis=2)
    opt_flow_mask = combo_mask < l_C

    return opt_flow_mask