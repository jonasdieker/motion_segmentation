# Motion Segmentation

The goal of the project was to segment the dynamic objects in a scene. The crux lies in disentangling what motion in the scene comes from ego vehicle transformation and what from actually dynamic objects. The resulting binary masks can then be used downstream for evaluating dynamic objects.

A major part of the contribution of this work is the generation of a dataset for the specific task of motion segmentation. As this is not a standard computer vision problem, limited ground truth data is available for training. A realistic multi-modal dataset was generated using the CARLA simulator by adjusting maps to include parked vehicles in addition to spawning a large number of moving vehicles and pedestrians.

Two distinc method were subsequently devised in order to perform motion segmentation on the created dataset. The first is a supervised method based on the [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) model with focal loss.

The second approach is an unsupervised method, where multiple coupled sub-problems are solved simultaneously to generate its own ground truth. Separate networks predict the depth and optical flow of the scene, which together can be used to compute the ground truth binary mask. Initially, both the optical flow and depth networks were simply simulated by using our multi-modal dataset with future extensions additionally learning these two data.

The project report can be found [here](final_report.pdf).

## Generated CARLA Dataset ([Docs](/docs/carla.md))

 - Explore our CARLA dataset in [Jupyter Notebook](/Carla/dataset_visualization.ipynb)

<p align="center">
  <img src=docs/assets/CarlaSamples.jpg>
</p>

## Supervised Motion Segmentation ([Docs](docs/supervised.md))

<p align="center">
  <img width=600px src=docs/assets/supervised_arch.png>
</p>

| Dataset      | IoU        | Aggregated IoU    | mIoU |
|:------------:|:----------:|:-----------------:|:----:|
| KITTI        | 0.749      | 0.730             | 0.871|
| CARLA        | 0.722      | 0.744             | 0.856|

 - Run training in [Jupyter Notebook](/supervised/train.ipynb)
 - Run inference in [Jupyter Notebook](/supervised/inference.ipynb)

## Unsupervised Motion Segmentation ([Docs](docs/unsupervised.md))

<p align="center">
  <img width=600px src=docs/assets/unsupervised_arch.png>
</p>

- Run training in [Jupyter Notebook](/unsupervised/train.ipynb)
