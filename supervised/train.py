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

# specify some hyperparams
lr = 5e-3
batch_size = 2
epochs = 100
alpha = 0.25
gamma = 2.0
print(f"running with lr={lr}, batch_size={batch_size}, epochs={epochs}")

# setup time/date for logging/saving models
now = datetime.now()
now_string = now.strftime(f"%d-%m-%Y_%H-%M_{batch_size}_{lr}_{epochs}")

# setup logging
log_root = "/storage/remote/atcremers40/motion_seg/logs"
logger = logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(log_root, f'{now_string}.log')),
        logging.StreamHandler(sys.stdout)
    ])

# set device and clean up
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

# data split and data loader
train_size = int(0.8 *  len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

# init model and pass to `device`
input_channels=6
output_channels=1
model = UNET(in_channels=input_channels, out_channels=output_channels).to(device)
# model = UNET_Mod(input_channels, output_channels).to(device)
model = model.float()

# for running single batch
# [data, targets] = next(iter(train_loader))
# print(len(np.where(targets[0] > 0)[0]))
# plt.imshow(targets[0].detach().cpu().permute(1,2,0).numpy())
# plt.show()

# loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

def run_val(loader, model):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device).float()
            y = y.to(device=device).float()

            # forward
            scores = model(x)
            loss = sigmoid_focal_loss(scores, y, alpha=alpha, gamma=gamma, reduction="sum")
            # loss = criterion(scores, y)
            val_losses.append(loss.item())

            if batch_idx == 0:
                sigmoid = nn.Sigmoid()
                scores_rounded = torch.round(sigmoid(scores))
                writer.add_images("visualised_preds", scores_rounded, global_step=epoch+1)
                writer.add_images("visualised_gts", y, global_step=epoch+1)

        writer.add_scalar("val loss", sum(val_losses)/len(val_losses), epoch*len(val_loader)) 

    # set back to train ensures layers like dropout, batchnorm are used after eval
    model.train()
    return sum(val_losses)/len(val_losses)


# initialise tensorboard
writer = SummaryWriter("/storage/remote/atcremers40/motion_seg/runs/" + now_string)

# for model saving
model_name_prefix = now.strftime("%d-%m-%Y")

# train network
print("train network ...")
train_loss = []
val_loss = []
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
    val_loss.append(run_val(val_loader, model))
    end = time.time()
    total_time += (end-start)
    scheduler.step(val_loss[epoch])
    logging.info(f"Epoch [{epoch + 1}/{epochs}] with lr {optimizer.param_groups[0]['lr']}, train loss: {round(train_loss[-1], 5)}, val loss: {round(val_loss[-1], 5)}, ETA: {round(((total_time/(epoch+1))*(epochs-epoch-1))/60**2,2)} hrs")

    save_path = f"/storage/remote/atcremers40/motion_seg/saved_models/{model_name_prefix}_{batch_size}_{lr}_{epoch}.pt"
    torch.save(model, save_path)

writer.close()

save_path = f"/storage/remote/atcremers40/motion_seg/saved_models/{model_name_prefix}_{batch_size}_{lr}_{epochs}.pt"
torch.save(model, save_path)