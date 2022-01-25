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

class KITTI_MOD_FIXED(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        img_dir = self.data_dir + 'images/'
        # return len(os.listdir(img_dir)) - 50
        return 100

    def __getitem__(self, idx):
        """
        Output: 
            Concatenated image [Bx6xHxW], Mask [Bx1xHxW]
        """

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
        else:
            img_path_1 = img_dir + img_dir_sort[idx+1]

        image_0 = np.array(Image.open(img_path_0), np.float32)
        image_1 = np.array(Image.open(img_path_1), np.float32)

        label_0 = torch.from_numpy(np.array(Image.open(label_path_0), np.float32))
        label_0 = label_0[None, :, :]

        if self.transform:
            img_0_tensor = self.transform(image_0)
            img_1_tensor = self.transform(image_1)
            if img_0_tensor.shape != torch.Size([3,375,1242]) or img_1_tensor.shape != torch.Size([3,375,1242]):
                print(img_path_0)
                return
            img_concat = torch.vstack([img_0_tensor, img_1_tensor])
        else:
            img_0_tensor = torch.from_numpy(image_0)
            img_1_tensor = torch.from_numpy(image_1)
            img_concat = torch.vstack([img_0_tensor, img_1_tensor])

        return (img_concat/255, label_0/255)


class ExtendedKittiMod(Dataset):
    def __init__(self, data_root, transform=None, test=False):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        # self.image_paths = sorted(list(glob.glob(os.path.join(self.data_root, "images/**/data/*.png"))))
        # self.mask_paths = sorted(list(glob.glob(os.path.join(self.data_root, "masks/**/image_02/*.png"))))

        if test == False:
            # wanted_dirs = ["0005", "0013", "0014", "0015", "0032", "0051", "0056", "0059", "0060"]
            # wanted_dirs = ["0013", "0014", "0015", "0018", "0056", "0057", "0059", "0060", "0084"]
            wanted_dirs = ["0014", "0015", "0018", "0032", "0056", "0057", "0059", "0060", "0084"]

        else:
            # wanted_dirs = ["0018", "0057", "0084"]
            # wanted_dirs = ["0005", "0032", "0051"]
            wanted_dirs = ["0005", "0013", "0051"]
        
        for sequence_num in wanted_dirs:
            self.image_paths.extend(sorted(list(glob.glob(os.path.join(self.data_root, f"images/2011_09_26_drive_{sequence_num}_sync/data/*.png")))))
            self.mask_paths.extend(sorted(list(glob.glob(os.path.join(self.data_root, f"masks/2011_09_26_drive_{sequence_num}_sync/image_02/*.png")))))

        temp_image_paths = self.image_paths.copy()
        temp_masks_paths = self.mask_paths.copy()

        dirs = []

        # removing final image in each sequence as it wont have a pair
        for i in range(len(self.image_paths)-1):
            path1 = self.image_paths[i]
            path2 = self.image_paths[i+1]
            if path1.split("/")[-3] != path2.split("/")[-3]:
                temp_image_paths.remove(path1)

            mask1 = self.mask_paths[i]
            mask2 = self.mask_paths[i+1]
            if mask1.split("/")[-3] != mask2.split("/")[-3]:
                temp_masks_paths.remove(mask1)

            if path1.split("/")[-3] not in dirs:
                dirs.append(path1.split("/")[-3])

            # if path1 in []:
            #     temp_image_paths.remove(path1)
            #     temp_masks_paths.remove(mask1)

        temp_image_paths = temp_image_paths[:-1]
        temp_masks_paths = temp_masks_paths[:-1]

        self.image_paths = temp_image_paths
        self.mask_paths = temp_masks_paths

        print(f"dirs loaded:\n{dirs}")

    def __len__(self):
        return len(self.image_paths)

    def get_pair_image(self, path):
        img_name = path.split("/")[-1]
        img_num = int(img_name.split(".")[0])
        pair_name = f"{img_num+1:010}.png"
        pair_path = os.path.join("/".join(path.split("/")[:-1]), pair_name)

        return pair_path

    def __getitem__(self, idx):

        image_0 = np.array(Image.open(self.image_paths[idx]), np.float32)
        image_1 = np.array(Image.open(self.get_pair_image(self.image_paths[idx])), np.float32)

        label_0 = torch.from_numpy(np.array(Image.open(self.mask_paths[idx]), np.float32))
        label_0 = label_0[None, :, :]
        
        if self.transform:
            img_0_tensor = self.transform(image_0)
            img_1_tensor = self.transform(image_1)
            img_concat = torch.vstack([img_0_tensor.permute((2,0,1)), img_1_tensor.permute((2,0,1))])
        else:
            img_0_tensor = torch.from_numpy(image_0)
            img_1_tensor = torch.from_numpy(image_1)
            img_concat = torch.vstack([img_0_tensor.permute((2,0,1)), img_1_tensor.permute((2,0,1))])

        # normalizing images so that each image channel (RGB) has a similar distribution
        return (img_concat/255, label_0/255)



"""
#Example

from torchvision import transforms
from torch.utils.data import DataLoader
data_dir = '/storage/remote/atcremers40/motion_seg/datasets/KITTI_MOD_fixed/training/'
data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
data = KITTI_MOD_FIXED_Dataset(data_dir, data_transforms)
dataloader= {
    'train': DataLoader(data, batch_size=1, shuffle=False)}
input, gt = next(iter(dataloader["train"]))
"""

def test():
    data_root = '/storage/remote/atcremers40/motion_seg/datasets/KITTI_MOD_fixed/training/'
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = KITTI_MOD_FIXED(data_root, data_transforms)
    item = dataset.__getitem__(0)
    print(f"len of dataset: {len(dataset)}\nshape of data: {item[0].shape}\nshape of targets: {item[1].shape}")

def test_ExtendedKittiMod():
    data_root = "/storage/remote/atcremers40/motion_seg/datasets/Extended_MOD_Masks/"
    dataset = ExtendedKittiMod(data_root)
    item = dataset.__getitem__(0)
    print(f"len of dataset: {len(dataset)}\nshape of data: {item[0].shape}\nshape of targets: {item[1].shape}")


if __name__ == "__main__":
    test_ExtendedKittiMod()
    #test()