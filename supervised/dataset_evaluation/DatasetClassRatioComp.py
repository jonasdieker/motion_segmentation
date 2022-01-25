"""
Data Loader for KITTI_MOD_FIXED as seen in 'MODNet: Moving Object Detection Network'
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import glob
import cv2

class ExtendedKittiMod(Dataset):
    def __init__(self, data_root, transform=None, test=False):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        if test == False:
            # wanted_dirs = ["0005", "0013", "0014", "0015", "0032", "0051", "0056", "0059", "0060"]
            # wanted_dirs = ["0013", "0014", "0015", "0018", "0056", "0057", "0059", "0060", "0084"]
            # wanted_dirs = ["0005", "0014", "0015", "0018", "0032", "0056", "0057", "0059", "0084"]
            wanted_dirs = ["0014", "0015", "0018", "0032", "0056", "0057", "0059", "0060", "0084"]

        else:
            # wanted_dirs = ["0018", "0057", "0084"]
            # wanted_dirs = ["0005", "0032", "0051"]
            # wanted_dirs = ["0013", "0051", "0060"]   # 2:3 (test:val)
            wanted_dirs = ["0005", "0013", "0051"]


        for sequence_num in wanted_dirs:
            self.image_paths.extend(sorted(list(glob.glob(os.path.join(self.data_root, f"images/2011_09_26_drive_{sequence_num}_sync/data/*.png")))))
            self.mask_paths.extend(sorted(list(glob.glob(os.path.join(self.data_root, f"masks/2011_09_26_drive_{sequence_num}_sync/image_02/*.png")))))

        temp_image_paths = self.image_paths.copy()
        temp_masks_paths = self.mask_paths.copy()

        dirs = []

        # removing final image in each sequence as it wont have a pair
        for i in range(len(self.image_paths)):
            if self.image_paths[i].split("/")[-3] not in dirs:
                dirs.append(self.image_paths[i].split("/")[-3])

        print(f"dirs loaded:\n{dirs}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # im = np.array(Image.open(self.image_paths[idx]), np.float32)

        im = cv2.imread(self.image_paths[idx])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        label_0 = torch.from_numpy(np.array(Image.open(self.mask_paths[idx]), np.float32))
        label_0 = label_0[None, :, :]

        # normalizing images so that each image channel (RGB) has a similar distribution
        return (im, label_0/255)


def test_ExtendedKittiMod():
    data_root = "/storage/remote/atcremers40/motion_seg/datasets/Extended_MOD_Masks/"
    dataset = ExtendedKittiMod(data_root)
    item = dataset.__getitem__(0)
    print(f"len of dataset: {len(dataset)}\nshape of data: {item[0].shape}\nshape of targets: {item[1].shape}")


if __name__ == "__main__":
    test_ExtendedKittiMod()