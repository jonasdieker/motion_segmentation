import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from skimage import io
import cv2

#------------------------- TRANSFORMATIONS ---------------------------#

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

#---------------------------- CV SPECIFIC STUFF ------------------------#

def read_depth(depth_file):
    depth = io.imread(depth_file)
    depth = depth[:, :, 0] * 1.0 + depth[:, :, 1] * 256.0 + depth[:, :, 2] * (256.0 * 256)
    depth = depth * (1/ (256 * 256 * 256 - 1))
    return depth

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
    # unpacking (christmas presents)
    u_coords, v_coords = u[:,0], u[:,1]

    # get homogeneous coords
    p = np.array([u_coords, v_coords, np.ones_like(u_coords)])

    # get K
    if K is None:
        K = get_intrinsics(72/(2*np.pi), image_size_x, image_size_y)

    # get 3D points
    p3d = np.dot(np.linalg.inv(K), p) * depth.reshape((-1)) * 1000
    return p3d.T

def transform_pointcloud(pc, trs):
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

def get_flow(depth1, trs, K=None):
    image_size_x = depth1.shape[1]
    image_size_y = depth1.shape[0]
    pixel_length = image_size_x * image_size_y
    u_coords = repmat(np.r_[image_size_x-1:-1:-1],
                        image_size_y, 1).reshape(pixel_length)
    v_coords = repmat(np.c_[image_size_y-1:-1:-1],
                        1, image_size_x).reshape(pixel_length)

    u_coords = np.flip(u_coords)
    v_coords = np.flip(v_coords)
    u = np.array([u_coords, v_coords]).T # u horizontal (x), v vertical (-y)

    pc1 = reproject(u, depth1, image_size_x, image_size_y, K)

    # (CX_T_1 @ 1_T_Sensor1)^-1 @ C2_trs_C1 @ (CX_T_1 @ 1_T_Sensor2)
    CX_T_Sensor = (roty(90) @ rotz(90))
    CX_T_Sensor_inv = inverse_se3(CX_T_Sensor)
    trs = CX_T_Sensor_inv @ trs @ CX_T_Sensor

    pc1_trs = transform_pointcloud(pc1, trs)
    u_dash = project(pc1_trs, image_size_x, image_size_y)

    flow = u_dash - u
    flow = flow.reshape(image_size_y, image_size_x, 2)

    return flow

# ------------------------- VISUALIZATIONS --------------------------------#

def plot_pair(img1, img2, size = (15,5)):
    plt.figure(figsize=size)
    plt.imshow(np.hstack((img1, img2)))
    plt.show()

def vis_flow(flow):
    # create HSV & make Value a constant
    hsv = np.zeros((512,1382,3))
    hsv[:,:,1] = 255

    # Encoding: convert the algorithm's output into Polar coordinate
    flow_reshaped = flow.reshape((512,1382,-1))
    mag, ang = cv2.cartToPolar(flow_reshaped[...,0], flow_reshaped[...,1])
    # Use Hue and Value to encode the Optical Flow
    hsv[:,:, 0] = ang * 180 / np.pi / 2
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    plt.figure(figsize=(20,10))
    plt.imshow(bgr)

def render_optical_flow_data(data):
    intensity = np.linalg.norm(data, axis=2)
    angle = np.arctan2(data[:, :, 0], data[:, :, 1])
    max_intensity = 100
    # N.B.: an intensity of exactly 1.0 makes the output black (perhaps showing the over-saturation), so keep it < 1
    intensity = np.clip(intensity, 0, max_intensity - 1) / max_intensity
    # log scaling
    basis = 30
    intensity = np.log1p((basis - 1) * intensity) / np.log1p(basis - 1)
    # for the angle they use 360° scale, see https://stackoverflow.com/a/57203497/14467327
    angle = (np.pi + angle) * 360 / (2 * np.pi)
    intensity = intensity[:, :, np.newaxis]
    angle = angle[:, :, np.newaxis]
    hsv_img = np.concatenate((angle, np.ones_like(intensity), intensity), axis=2)
    img_out = np.array(cv2.cvtColor(np.array(hsv_img, dtype=np.float32), cv2.COLOR_HSV2RGB) * 256,
                       dtype=np.dtype("uint8"))
    plt.figure(figsize=(20,10))
    plt.imshow(img_out)

def plot_sparse_vecs(flow, sparseness=30, cutoff=1):    
    fig = plt.figure(figsize = (16,10))
    plt.xlim(0,flow.shape[1])
    plt.ylim(0,flow.shape[0])
    plt.gca().invert_yaxis()

    disc_step = sparseness
    of_offset = 10

    for i in range(of_offset, flow.shape[0]-of_offset, disc_step):
        for j in range(of_offset,flow.shape[1]-of_offset, disc_step):
            if np.linalg.norm(flow[i,j,:]) > cutoff:
                plt.arrow(j,i,flow[i,j,0], flow[i,j,1], head_width=5)
    fig.show()
