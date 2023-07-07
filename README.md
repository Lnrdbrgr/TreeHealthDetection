# TreeHealthDetection

> Ongoing project. Description, methodology and code base can be subject to changes at any time.

## About the Project

#### Background
The irreversible consequences of man-made climate change will affect our life and that of all other species and nature permanently from now on. We will have to live and deal with the consequences of global warming. For that, it is crucial to limit the damage and destruction of nature and our surroundings. 

One consequence of the temperature increase that is already visible today is the increased ability of pest insects to reproduce, such as the European spruce bark beetle (Ips typographus L.). In Europe, the bark beetle infests spruce trees and leads to rapid defoliation with certain death of the tree. The beetle colony then inhabits surrounding trees, leaving rotten forests and wastelands. Identifying infested trees as early as possible is crucial in order to remove those and prevent spreading.

This project aims at classifying infested spruce trees from aerial imagery, a method to make forest management and maintenance more efficient and providing the ability to react fast when necessary.

#### Methodology
The health status is classified using RCNNs (Regional Convolutional Neural Networks) at tree level by providing bounding box predictions along with the classification status. Image data is collected in 5 areas with ongoing infestation using a DJI drone over the spring and summer of 2023. Image data is not shared in this repository but some examples along with bounding boxes can be found in Data/GitHubImages to get the pipeline running (tbd).

## Code

The project is divided in different modules represented by the folders above. This repository provides an End-to-End solution from the initial orthomosaic to a trained model along with accuracy measures. The code is kept as generall as possible to enable object detection on any custom dataset.

Pre-processing scripts can be found under PreProcessing. The scripts expect a (large tif-) image along with a csv of bounding box coordinates and classification status. The large image is divided to smaller subsamples to enable model training. For each smaller image, an xml in Pascal-VOC style is generated to store bounding box information.

Scripts for model training are located under Detection. engine.py runs the model training. The remaining scripts collect utility and helper functions and classes. Image augmentation transformations can be set in transformations.py.

