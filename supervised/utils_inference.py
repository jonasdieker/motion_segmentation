import numpy as np
import torch
from torch.utils.data import DataLoader
from DatasetClass import KITTI_MOD_FIXED, ExtendedKittiMod, CarlaMotionSeg
from torch.nn import Sigmoid
import matplotlib.pyplot as plt


def get_dataloaders(dataset, batch_size=2, dataset_fraction=1.0, shuffle=True):
    dataset_length = int(len(dataset) * dataset_fraction)
    train_size = int(0.6 *  dataset_length)
    val_size = int(0.5*(dataset_length - train_size))
    test_size = dataset_length - train_size - val_size
    # taking subset of dataset according to dataset_fraction
    dataset_idx = list(range(0, dataset_length))
    dataset = torch.utils.data.Subset(dataset, dataset_idx)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader


def load_model(model_path, device='cuda:0'):
    if device == "cuda:0":
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            device = torch.device(device)
            print("Moving model to GPU")
            return torch.load(model_path).to(device)
        else:
            print("No GPU available!")

    if device == "cpu":
        print("Model on CPU")
        return torch.load(model_path, map_location=lambda storage, loc: storage)


def calc_iou_moving(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    
    intersection = torch.sum(torch.bitwise_and(outputs, labels).float(), (1,2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum(torch.bitwise_or(outputs, labels).float(), (1,2))          # Will be zero if both are 0
    
    IoU = (intersection + SMOOTH) / (union + SMOOTH)   # avoid division by zero
    IoU = IoU.mean()
        
    return IoU

def calc_iou_background(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)

    outputs = (outputs -1) * -1
    labels = (labels -1) * -1
    
    intersection = torch.sum(torch.bitwise_and(outputs, labels).float(), (1,2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum(torch.bitwise_or(outputs, labels).float(), (1,2))          # Will be zero if both are 0
    
    IoU = (intersection + SMOOTH) / (union + SMOOTH)   # avoid division by zero
    IoU = IoU.mean()
        
    return IoU

def mIoU(outputs: torch.Tensor, labels: torch.Tensor):
    iou_moving = calc_iou_moving(outputs, labels)
    iou_background = calc_iou_background(outputs, labels)
    mIoU = (iou_moving + iou_background)/2
    return mIoU


def confusion_matrix(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1)

    TP = torch.sum(torch.bitwise_and(outputs, labels).float(), (1,2)) # intersection
    FP = torch.sum(outputs,(1,2)) - TP
    FN = torch.sum(labels, (1,2)) - TP
    TN = torch.sum(labels, (1,2)) - torch.sum(torch.bitwise_or(outputs, labels).float(), (1,2)) # union

    return TP, FP, TN, FN

def aggregated_IoU(outputs, labels):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1)
    
    intersection = torch.sum(torch.bitwise_and(outputs, labels).float(), (1,2))  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum(torch.bitwise_or(outputs, labels).float(), (1,2))          # Will be zero if both are 0
    
    agg_IoU = (torch.sum(intersection, 0) + SMOOTH) / (torch.sum(union, 0) + SMOOTH)   # avoid division by zero
    return agg_IoU

   
def inference_prep(dataset_type="carla", model_type="carla", dataset_fraction=1.0, device="cuda:0"):
    if dataset_type == "carla":
        data_root = "/storage/remote/atcremers40/motion_seg/datasets/Carla_supervised/"
        # data_root = "/storage/remote/atcremers40/motion_seg/datasets/other/Opt_flow_pixel_preprocess"
        dataset = CarlaMotionSeg(data_root)
    elif dataset_type == "kitti":
        data_root = "/storage/remote/atcremers40/motion_seg/datasets/Extended_MOD_Masks/"
        dataset = ExtendedKittiMod(data_root)
    if model_type == "carla":
        print("Loading carla model")
        model_path = "/storage/remote/atcremers40/motion_seg/saved_models/02-02-2022_08-21_bs2/best_aIoU.pt"   # CARLA trained
        # model_path = "/storage/remote/atcremers40/motion_seg/saved_models/10-03-2022_19-42_bs2/best_aIoU.pt" # Carla BCE
        # model_path = "/storage/remote/atcremers40/motion_seg/saved_models/13-03-2022_22-04_bs2/best_IoU.pt"
    elif model_type == "kitti":
        print("Loading kitti model")
        model_path = "/storage/remote/atcremers40/motion_seg/saved_models/01-02-2022_18-30_bs2/best_aIoU.pt"  # KITTI trained
        # model_path = "/storage/remote/atcremers40/motion_seg/saved_models/12-03-2022_08-28_bs2/best_aIoU.pt" # KITTI BCE

    train_loader, val_loader, test_loader = get_dataloaders(dataset, dataset_fraction=dataset_fraction)
    model = load_model(model_path, device=device)
    return model, train_loader, val_loader, test_loader

def run_inference(model, loader, metric="IoU", device="cpu"):
    sigmoid = Sigmoid()
    device = torch.device(device)
    with torch.no_grad():
        model.eval()
        test_iou = []
        for idx, (data, targets) in enumerate(loader):
            data = data.to(device).float()
            targets = targets.to(device).int()

            # forward
            scores = model(data)
            scores_rounded = torch.round(sigmoid(scores)).int()
            if metric == "IoU":
                test_iou.append(calc_iou_moving(scores_rounded, targets))
            elif metric == "mIoU":
                test_iou.append(mIoU(scores_rounded, targets))
            elif metric == "AggIoU":
                test_iou.append(aggregated_IoU(scores_rounded, targets))

    print(f"{metric}: {sum(test_iou)/len(test_iou)}")

#--------------------------- VISUALIZE ---------------------------#

def vis_batch(scores, data, targets):
    # pass to CPU for visualization
    scores = scores.detach().cpu()
    targets = targets.detach().cpu().int()
    data = data.detach().cpu()
    sigmoid = Sigmoid()
    scores_rounded = torch.round(sigmoid(scores)).int()

    print(f"First IoU: {calc_iou_moving(scores_rounded[0], targets[0])}")
    print(f"Second IoU: {calc_iou_moving(scores_rounded[1], targets[1])}")

    # plotting
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(30, 8))
    ax1.imshow(targets[0][0])
    ax1.set_title("gt1")
    ax2.imshow(scores_rounded.detach().numpy()[0][0])
    ax2.set_title("scores1")
    ax3.imshow(np.transpose(data[0][:3], axes=[1, 2, 0]))
    ax3.set_title("data1")
    ax4.imshow(targets[1][0])
    ax4.set_title("gt2")
    ax5.imshow(scores_rounded.detach().numpy()[1][0])
    ax5.set_title("scores2")
    ax6.imshow(np.transpose(data[1][:3], axes=[1, 2, 0]))
    ax6.set_title("data2")
    plt.show()