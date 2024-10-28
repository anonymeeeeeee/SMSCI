# SMSCI: Simultaneous Modeling of Social and Contextual Interactions for Multi Pedestrian Trajectory Prediction

This repository contains the implementation in Pytorch of SMSCI: Simultaneous Modeling of Social and Contextual Interactions for Multi Pedestrian Trajectory Prediction.

<div align="center">
  <img src="https://github.com/anonyme-anonymee/SMSCI/assets/159822306/5836d42a-3ae2-4e8d-a583-58d5c9ac9eee" alt="zara2_gif" width="300">
</div>


# Overview

We explore the limitations of prior methods reliant on pedestrian positions for modeling social interactions and scene influences. 
This research achieves a breakthrough in simultaneously predicting multiple pedestrian trajectories in a scene video by introducing the SMSCI model which uses Generative Adversarial Networks and sequence prediction. 
By integrating both historical paths and environmental context, SMSCI accounts for both social interactions and scene influences, outperforming previous approaches. 
Through experiments across various datasets demonstrating enhanced accuracy and collision avoidance, the effectiveness of generating socially and physically valid trajectories is demonstrated. 

![overview](https://github.com/anonyme-anonymee/SMSCI/assets/159822306/cc0361b7-2e20-4060-a6c5-b7027c948813)

# Prerequisites

To install all the dependency packages, please run:

```
pip install -r requirements.txt
```

# Data Preparation

We use the following datasets : ETH, UCY, SDD

1- Please, run the following script to download the ETH / UCY datasets:

```bash
bash scripts/download_data.sh
```

This will create the directory `datasets/<dataset_name>` with train/ val/ and test/ splits. All the datasets are pre-processed to be in world coordinates i.e. in meters. We support five datasets ETH, ZARA1, ZARA2, HOTEL and UNIV. 
We use leave-one-out approach, train on 4 sets and test on the remaining set. 

For Stanford Drone dataset (SDD), each video for each scene in the videos directory has an associated annotation file (annotation.txt) and exemplary frame (reference.jpg) in the annotations directory. It consists of annotated videos of pedestrians, bikers, skateboarders, cars, buses, and golf carts navigating eight unique scenes on the Stanford University campus.

Please download SDD dataset from their website (https://cvgl.stanford.edu/projects/uav_data/).

Annotation file format:
Each line in the annotations.txt file corresponds to an annotation. Each line contains 10+ columns, separated by spaces. The definition of these columns are:

    1   Track ID. All rows with the same ID belong to the same path.
    2   xmin. The top left x-coordinate of the bounding box.
    3   ymin. The top left y-coordinate of the bounding box.
    4   xmax. The bottom right x-coordinate of the bounding box.
    5   ymax. The bottom right y-coordinate of the bounding box.
    6   frame. The frame that this annotation represents.
    7   lost. If 1, the annotation is outside of the view screen.
    8   occluded. If 1, the annotation is occluded.
    9   generated. If 1, the annotation was automatically interpolated.
    10  label. The label for this annotation, enclosed in quotation marks.

As in most other research that using SDD for trajectory prediction as PCENet (ECCV 2020), YNet (ICCV 2021), in our data processing, we only keep the trajectories with label "Pedestrian", where we keep all the trajectories.

We observe the trajectory for 8 times steps (3.2 seconds) and show prediction results for 8 (3.2 seconds) and 12 (4.8 seconds) time steps.

2- Set the path for saving your trained models in the code.

3- Run the following scripts (train and evaluate.py) under scripts to train and test the model.

# Notes
The repository is still under construction. Please let me know if you encounter any issues.

Best, 
