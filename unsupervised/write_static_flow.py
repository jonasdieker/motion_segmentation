import numpy as np
import glob
import json
import os
from tqdm import tqdm

from utils_data import get_flow, read_depth


if __name__ == "__main__":

    print("writing static flow...")

    data_root = "/storage/remote/atcremers40/motion_seg/datasets/CARLA/"
    depth_root = os.path.join(data_root, "depth")
    trs_root = os.path.join(data_root, "transformations")
    static_flow_root = os.path.join(data_root, "static_flow")

    depth_sequences = sorted(list(glob.glob(os.path.join(depth_root, "**"))))
    trs_sequences = sorted(list(glob.glob(os.path.join(trs_root, "**"))))
    # iterate over sequences
    for seq in tqdm(range(len(depth_sequences))):

        depths = sorted(list(glob.glob(os.path.join(depth_sequences[seq], "*.png"))))
        transformations = os.path.join(trs_sequences[seq], "transforms.json")

        with open(transformations, "r") as f:
            trs_list = json.load(f)["transforms"]

        if not os.path.isdir(static_flow_root):
            os.mkdir(static_flow_root)

        sequence_num = str(depth_sequences[seq]).split("/")[-1]
        sequence_path = os.path.join(static_flow_root, sequence_num)
        if not os.path.isdir(sequence_path):
            os.mkdir(sequence_path)

        for frame_idx in range(1,len(depths)):

            depth = read_depth(depths[frame_idx])
            trs = np.array(trs_list[frame_idx])
            flow = get_flow(depth, trs)

            static_flow_path = os.path.join(sequence_path, "%04d.pkl" %(frame_idx))
            with open(static_flow_path, "wb") as f:
                np.save(f, np.array(flow))

