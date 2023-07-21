"""Module to create different object detection models.
"""

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
from typing import Any


def create_FasterRCNN_resnet50_model(num_classes: int) -> Any:
    """
    Creates a FasterRCNN Model with pre-trained ResNet weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem.

    Returns:
        (Any):
            The Model.
    """
    # load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    # get number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features,
                                                      num_classes=num_classes+1)
    # return
    return model


def create_FasterRCNN_mobilenet_v3_model(num_classes: int) -> Any:
    """
    Creates a FasterRCNN Model with pre-trained mobilenet weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem.

    Returns:
        (Any):
            The Model.
    """
    # load pre-trained model
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT')
    # get number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features,
                                                      num_classes=num_classes+1)
    # return
    return model


def create_MaskRCNN_resnet50_model(num_classes: int) -> Any:
    """
    Creates a MaskRCNN Model with pre-trained resnet50 weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem.

    Returns:
        (Any):
            The Model.
    """
    # load pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='MaskRCNN_ResNet50_FPN_Weights.DEFAULT')
    # get number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features,
                                                      num_classes=num_classes+1)
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask,
                                                       num_classes=num_classes)
    # return
    return model
