# TreeHealthDetection

This project is part of my Master Thesis "Evaluating Forest Health: A Deep Neural Network Approach to Detect and Forecast Bark Beetle Infestation in Spruce Trees" at the University of Tübingen, submitted 11.12.2023. Images are taken from the thesis.

The irreversible consequences of man-made climate change will affect our life and that of all other species and nature permanently from now on. We will have to live and deal with the consequences of global warming. For that, it is crucial to limit the damage and destruction of nature and our surroundings. 

One consequence of the temperature increase already visible today is the increased ability of pest insects to reproduce, such as the European spruce bark beetle (Ips typographus L.). In Europe, the bark beetle infests spruce trees and leads to rapid defoliation with certain death of the tree. The beetle colony then inhabits surrounding trees, leaving rotten forests and wastelands. Identifying infested trees as early as possible is crucial in order to remove those and prevent spreading.

![image](https://github.com/Lnrdbrgr/TreeHealthDetection/assets/78366819/b540abe1-a45d-40e3-833c-3e825f4cac14)

This project aims at classifying infested spruce trees from aerial imagery, a method to make forest management and maintenance more efficient and providing the ability to react fast when necessary. Four deep-learning based object detection algorithms are used to identify infestation status of spruce trees in five different locations across Germany. 

Detection and classification was reached with F1-Scores of up to 0.63 and 0.59 on average across various locations.

![image](https://github.com/Lnrdbrgr/TreeHealthDetection/assets/78366819/a7238455-be50-481c-8f44-8f280ad3d31c)


# Working with the Code

The code consists of two main scripts for model training (train_engine_detection.py and train_engine_segmentation.py) that can be run from the command line with the specified configurations. The training scripts create an output folder with corresponding models and training details. Once the training is finished, inference scripts (inference_detection.py and inference_segmentation.py) can be run to evaluate the model on the test set. Data is not shared in this repository but can be made available under certain conditions. Please reach out for comments and discussions!


