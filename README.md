# ws21_motionsegmentation

The aim of the project lies in training a model that produces accurate binary segmentation of moving objects in autonomous driving scenario.

The final report for the project can be found [here](final_report.pdf).

## Generated CARLA Dataset ([Docs](/docs/carla.md))

 - Explore our CARLA dataset in [Jupyter Notebook](/Carla/data_visualization.ipynb)

![Collection](/docs/assets/CarlaSamples.jpg)

## Supervised Motion Segmentation ([Docs](docs/supervised.md))

| Dataset      | IoU        | Aggregated IoU    | mIoU |
|:------------:|:----------:|:-----------------:|:----:|
| KITTI        | 0.749      | 0.730             | 0.871|
| CARLA        | 0.722      | 0.744             | 0.856|

*insert example images here*

 - Run training in [Jupyter Notebook](/supervised/train.ipynb)
 - Run inference in [Jupyter Notebook](/supervised/inference.ipynb)

## Unsupervised Motion Segmentation ([Docs](docs/unsupervised.md))

| Dataset      | IoU        |
|:------------:|:----------:|
| KITTI        | 0.xxx      |
| CARLA (own)  | 0.xxx      |

*insert example images here*

- Run training in [Jupyter Notebook](/unsupervised/train.ipynb)