"""
Data Loader for KITTI_MOD_FIXED as seen in 'MODNet: Moving Object Detection Network'
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms

import os

class KITTI_MOD_FIXED_Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        img_dir = self.data_dir + 'images/'
        return len(os.listdir(img_dir))

    def __getitem__(self, idx):

        img_dir = self.data_dir + 'images/'
        label_dir = self.data_dir + 'mask/'

        img_dir_sort = sorted(os.listdir(img_dir))
        img_path_0 = img_dir + img_dir_sort[idx]
        label_path_0 = label_dir + img_dir_sort[idx]

        #Different set lengths for training vs testing
        testing = os.path.basename(self.data_dir) == 'testing'
        if testing:
            idx_list = [152, 348]
        else:
            #Training
            idx_list = [106, 182, 295, 564, 924, 1099, 1299]

        #In order to avoid using pairs from different sets, will use the previous img
        if idx in idx_list:
            img_path_1 = img_dir + img_dir_sort[idx-1]
            label_path_1 = label_dir + img_dir_sort[idx-1]

        else:
            img_path_1 = img_dir + img_dir_sort[idx+1]
            label_path_1 = label_dir + img_dir_sort[idx+1]

        image_0 = np.array(Image.open(img_path_0), np.float32)
        image_1 = np.array(Image.open(img_path_1), np.float32)

        label_0 = torch.from_numpy(np.array(Image.open(label_path_0), np.float32))
        label_1 = torch.from_numpy(np.array(Image.open(label_path_1), np.float32))

        if self.transform:
            img_0_tensor = self.transform(image_0)
            img_1_tensor = self.transform(image_1)
        else:
            img_0_tensor = torch.from_numpy(image_0)
            img_1_tensor = torch.from_numpy(image_1)

        return [img_0_tensor, img_1_tensor], [label_0, label_1] 


"""
Example
data_dir = '/storage/remote/atcremers40/motion_seg/datasets/KITTI_MOD_fixed/training/'
data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
data = KITTI_MOD_FIXED_Dataset(data_dir, data_transforms)
dataloader= {
    'train': DataLoader(data, batch_size=1, shuffle=False)}
input, gt = next(iter(dataloader["train"]))

"""