import sys
import os
import logging
from numpy.matlib import repmat

import torch
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

def get_dataloaders(dataset, args, shuffle=True):
    dataset_length = int(len(dataset) * args.dataset_fraction)
    train_size = int(0.6 *  dataset_length)
    args.train_size = train_size
    val_size = int(0.5*(dataset_length - train_size))
    test_size = dataset_length - train_size - val_size
    # taking subset of dataset according to dataset_fraction
    dataset_idx = list(range(0, dataset_length))
    dataset = torch.utils.data.Subset(dataset, dataset_idx)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader

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