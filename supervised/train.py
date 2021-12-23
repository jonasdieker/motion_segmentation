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
import sys
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import PIL

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

# specify some hyperparams
lr = 5e-3
batch_size = 4
epochs = 100
print(f"running with lr={lr}, batch_size={batch_size}, epochs={epochs}")

# data split and data loader
train_size = int(0.8 *  len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

# init model and pass to `device`
input_channels=6
output_channels=1
# model = UNET(in_channels=input_channels, out_channels=output_channels).to(device)
model = UNET_Mod(input_channels, output_channels).to(device)
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

def run_val(loader, model):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device=device).float()
            y = y.to(device=device).float()

            # forward
            scores = model(x)
            loss = sigmoid_focal_loss(scores, y, reduction="sum")
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
now = datetime.now()
now_string = now.strftime("%d.%m.%Y_%H:%M:%S")
writer = SummaryWriter("/storage/remote/atcremers40/motion_seg/runs/" + now_string)

# train network
print("train network ...")
train_loss = []
val_loss = []
for epoch in range(epochs):
    model.train()
    losses = []
    steps_per_epoch = len(train_loader)

    for batch_idx, (data, targets) in enumerate(train_loader):

        # move data to gpu if available
        data = data.to(device).float()
        targets = targets.to(device).float()

        # forward
        scores = model(data)
        loss = sigmoid_focal_loss(scores, targets, reduction="sum")
        # loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # adam step
        optimizer.step()

        losses.append(loss.item())

        if (batch_idx + 1) % 20 == 0:
            writer.add_scalar("training loss", sum(losses)/len(losses), epoch*steps_per_epoch + batch_idx)

    # print(f"Epoch {epoch}: loss => {sum(losses)/len(losses)}")
    train_loss.append(sum(losses)/len(losses))
    val_loss.append(run_val(val_loader, model))
    print(f"epoch [{epoch + 1}/{epochs}], train loss: {round(train_loss[-1], 5)}, val loss: {round(val_loss[-1], 5)}")

writer.close()