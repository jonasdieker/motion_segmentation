"""
Data Loader for Unsupervised Network using CARLA gt
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import glob


class CarlaUnsupervised(Dataset):
    """
    Dataset for unsupervised training which has 
     - RGB images (just for visualization not for network predictions)
     - Motion segmentation images
     - Optical flow from CARLA sensor
     - Pixelwise static optical flow vectors

    Network will predict motion segmentation
    """
    def __init__(self, data_root, test = False, transform=None):
        self.data_root = data_root
        self.test = test
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []
        self.depth_masks = []
        self.static_flow = []
        self.dynamic_flow = []
        
        image_seqs = (sorted(list(glob.glob(os.path.join(self.data_root, "images/**/")))))
        mask_seqs = (sorted(list(glob.glob(os.path.join(self.data_root, "motion_segmentation/**/")))))
        static_flow_seqs = (sorted(list(glob.glob(os.path.join(self.data_root, "static_flow/**/")))))
        dynamic_flow_seqs = (sorted(list(glob.glob(os.path.join(self.data_root, "opt_flow/**/")))))

        self.sequence_number = len(image_seqs)

        for seq in range(self.sequence_number):
            self.image_paths.extend(sorted(list(glob.glob(os.path.join(image_seqs[seq], "*.png"))))[10:-1])
            self.mask_paths.extend(sorted(list(glob.glob(os.path.join(mask_seqs[seq], "*.png"))))[10:-1])
            self.static_flow.extend(np.load(os.path.join(static_flow_seqs[seq], "static_flow.pkl"))[10:-1])
            self.dynamic_flow.extend(np.load(os.path.join(dynamic_flow_seqs[seq], "opt_flow.pkl"))[10:-1])
        #TODO: Ensure all files are loaded, otherwise print/log erros

    def __len__(self):
        image_path_len = len(self.image_paths)
        return image_path_len

    def get_pair_image(self, path):
        img_name = path.split("/")[-1]
        img_num = int(img_name.split(".")[0])
        pair_name = f"{img_num+1:04}.png"
        pair_path = os.path.join("/".join(path.split("/")[:-1]), pair_name)
        return pair_path

    def __getitem__(self, idx):

        image_0 = np.array(Image.open(self.image_paths[idx]), np.float32)[:,:,:3]
        image_1 = np.array(Image.open(self.get_pair_image(self.image_paths[idx])), np.float32)[:,:,:3]

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

        #Return rgb image, dynamic_opt_flow array, static_opt_flow_array, motion_mask 
        if self.test:
            return (img_concat/255, self.dynamic_flow[idx], self.static_flow[idx], label_0/255)
        ##Return rgb image, dynamic_opt_flow array, static_opt_flow_array
        else:
            return (img_concat/255, self.dynamic_flow[idx], self.static_flow[idx])


def test_Carla():
    data_root = "/storage/remote/atcremers40/motion_seg/datasets/Carla_tmp/"
    dataset = CarlaUnsupervised(data_root)
    item = dataset.__getitem__(0)
    print(f"len of dataset: {len(dataset)}\n \
    shape of rgb: {item[0].shape}\n \
    shape of dynamic flow: {item[1].shape}\n \
    shape of static flow: {item[2].shape}")


if __name__ == "__main__":
    test_Carla()