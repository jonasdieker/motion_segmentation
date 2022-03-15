# ws21_motionsegmentation

The aim of the project lies in training a model that produces accurate binary segmentation of moving objects in autonomous driving scenario.

The final report for the project can be found [here](final_report.pdf).

## CARLA Dataset Generation ([Docs](/docs/carla.md))

![Collection](/docs/assets/CarlaSamples.jpg)

 - Explore [CARLA dataset](/)

## Supervised Motion Segmentation ([Docs](docs/supervised.md))

| Dataset      | IoU        | Aggregated IoU    | mIoU |
|:------------:|:----------:|:-----------------:|:----:|
| KITTI        | 0.749      | 0.730             | 0.871|
| CARLA        | 0.722      | 0.744             | 0.856|

*insert example images here*

 - Run training interactively in [Jupyter Notebook](/)
 - Run inference interactively in [Jupyter Notebook](/supervised/inference.ipynb)

## Unsupervised Motion Segmentation ([Docs](docs/unsupervised.md))

| Dataset      | IoU        |
|:------------:|:----------:|
| KITTI        | 0.xxx      |
| CARLA (own)  | 0.xxx      |

*insert example images here*