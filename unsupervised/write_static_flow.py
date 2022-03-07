import numpy as np
from numpy.matlib import repmat
import glob
import matplotlib.pyplot as plt
import json
from skimage import io
import cv2
import os

def get_intrinsics(fov, image_size_x, image_size_y):
    f = image_size_x/(2.0 * np.tan(fov * np.pi /360))
    Cx = image_size_x / 2.0
    Cy = image_size_y / 2.0
    return np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]])

def reproject(u, depth, image_size_x, image_size_y, K=None):
    '''
    get [u,v] pixel coordinates and convert to homogeneous
    get instrinsics
    multiply inverse K with the homogeneous points and scale depth
    '''
    # unpacking
    u_coords, v_coords = u[:,0], u[:,1]

    # get homogeneous coords
    p = np.array([u_coords, v_coords, np.ones_like(u_coords)])

    # get K
    if K is None:
        K = get_intrinsics(72/(2*np.pi), image_size_x, image_size_y)

    # get 3D points
    p3d = np.dot(np.linalg.inv(K), p) * depth.reshape((-1)) * 1000
    return p3d.T

def rotx(angle):
    return np.array([
        [1,0,0,0],
        [0,np.cos((angle*np.pi)/180), -np.sin((angle*np.pi)/180), 0],
        [0,np.sin((angle*np.pi)/180), np.cos((angle*np.pi)/180), 0],
        [0,0,0,1]
    ])

def roty(angle):
    return np.array([
        [np.cos((angle*np.pi)/180),0, np.sin((angle*np.pi)/180),0],
        [0,1,0,0],
        [-np.sin((angle*np.pi)/180),0, np.cos((angle*np.pi)/180),0],
        [0,0,0,1]
    ])

def rotz(angle):
    return np.array([
        [np.cos((angle*np.pi)/180), -np.sin((angle*np.pi)/180),0,0],
        [np.sin((angle*np.pi)/180), np.cos((angle*np.pi)/180),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])

def build_se3(rotation, translation):
    se3 = np.hstack((rotation, np.array([translation]).T))
    se3 = np.vstack((se3,np.array([0,0,0,1])))
    return se3

def inverse_se3(se3_mat):
    R_T = se3_mat[:3, :3].T
    new_t = -R_T.dot(se3_mat[:3,-1])

    return build_se3(R_T, new_t)

def transform_pointcloud(pc, trs):
    # somehow ensure correct direction of trs
    pc = np.hstack([pc, np.ones((pc.shape[0],1))])
    pc_trs = trs@pc.T
    return pc_trs[:3,:].T

def project(p3d, image_size_x, image_size_y):
    '''
    gets intrinsics, projects points into the image plane
    and normalises the pixels
    '''   
    K = get_intrinsics(72/(2*np.pi), image_size_x, image_size_y)
    unnormalised_pixel_coords = np.dot(K, p3d.T).T
    pixel_coords = unnormalised_pixel_coords/(unnormalised_pixel_coords[:,2].reshape((-1,1)))
    return pixel_coords[:,:2]

def get_flow(depth0, image_size_x, image_size_y, trs, K=None):
    # 2d pixel coordinates
    pixel_length = image_size_x * image_size_y
    u_coords = repmat(np.r_[image_size_x-1:-1:-1],
                        image_size_y, 1).reshape(pixel_length)
    v_coords = repmat(np.c_[image_size_y-1:-1:-1],
                        1, image_size_x).reshape(pixel_length)

    u_coords = np.flip(u_coords)
    v_coords = np.flip(v_coords)
    u = np.array([u_coords, v_coords]).T # u horizontal (x), v vertical (-y)

    pc0 = reproject(u, depth0, image_size_x, image_size_y, K)

    # find transformation
    trs = inverse_se3(trs)

    # (CX_T_1 @ 1_T_Sensor1)^-1 @ C2_trs_C1 @ (CX_T_1 @ 1_T_Sensor2)
    CX_T_Sensor = (roty(90) @ rotz(90))
    CX_T_Sensor_inv = inverse_se3(roty(90) @ rotz(90))
    trs = CX_T_Sensor_inv @ trs @ CX_T_Sensor

    pc0_trs = transform_pointcloud(pc0, trs)
    u_dash = project(pc0_trs, image_size_x, image_size_y)

    flow = u_dash - u
    flow = flow.reshape(image_size_y, image_size_x, 2)

    return flow

def vis_flow(opt_flow):
    # create HSV & make Value a constant
    hsv = np.zeros((512,1382,3))
    hsv[:,:,1] = 255

    # Encoding: convert the algorithm's output into Polar coordinate
    opt_flow_reshaped = opt_flow.reshape((512,1382,-1))
    mag, ang = cv2.cartToPolar(opt_flow_reshaped[...,0], opt_flow_reshaped[...,1])
    # Use Hue and Value to encode the Optical Flow
    hsv[:,:, 0] = ang * 180 / np.pi / 2
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    plt.figure(figsize=(20,10))
    plt.imshow(bgr)

if __name__ == "__main__":

    print("writing static flow...")

    data_root = "/storage/remote/atcremers40/motion_seg/datasets/Carla_unsupervised/"
    depth_root = os.path.join(data_root, "depth")
    trs_root = os.path.join(data_root, "transformations")
    static_flow_root = os.path.join(data_root, "static_flow")

    depth_sequences = sorted(list(glob.glob(os.path.join(depth_root, "**"))))
    trs_sequences = sorted(list(glob.glob(os.path.join(trs_root, "**"))))
    # iterate over sequences
    for seq in range(len(depth_sequences)):

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

        for frame_idx in range(len(depths)-1):

            depth = plt.imread(depths[frame_idx])
            trs = np.array(trs_list[frame_idx+1])
            flow = get_flow(depth, 1382, 512, trs)

            static_flow_path = os.path.join(sequence_path, "%04d.pkl" %(frame_idx))
            with open(static_flow_path, "wb") as f:
                np.save(f, np.array(flow))

