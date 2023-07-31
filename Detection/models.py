"""Module to create different object detection models.
"""

from functools import partial
import math
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import torch
import torchvision
from typing import Any


def create_FasterRCNN_resnet50_model(num_classes: int) -> torch.nn.Module:
    """
    Creates a FasterRCNN Model with pre-trained ResNet weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem (without
            background class).

    Returns:
        model (torch.nn.module):
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


def create_FasterRCNN_mobilenet_v3_model(num_classes: int) -> torch.nn.Module:
    """
    Creates a FasterRCNN Model with pre-trained Mobilenet weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem (without
            background class).

    Returns:
        model (torch.nn.module):
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


def create_Retinanet_resnet50_v2_model(num_classes: int) -> torch.nn.Module:
    """
    Creates a RetinaNet Model with pre-trained ResNet weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem (without
            background class).

    Returns:
        model (torch.nn.module):
            The Model.
    """
    # load pre-trained model
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights='RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT'
    )
    # replace classification layer 
    num_anchors = model.head.classification_head.num_anchors
    in_features = 256
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = (num_classes+1)
    cls_logits = torch.nn.Conv2d(
        in_features, num_anchors * (num_classes+1),
        kernel_size = 3, stride=1, padding=1
    )
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits
    # return
    return model


def create_ssd300_vgg16_model(num_classes: int) -> torch.nn.Module:
    """
    Creates a Single Shot MultiBox Detector (SSD) Model with
    pre-trained VGG16 weights.

    Args:
        num_classes (int):
            Number of classes in the detection problem (without
            background class).

    Returns:
        model (torch.nn.module):
            The Model.
    """
    # load pre-trained model
    model = torchvision.models.detection.ssd300_vgg16(
        weights='SSD300_VGG16_Weights.DEFAULT'
    )
    # replace classification layer
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    norm_layer  = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes+1, norm_layer
    )
    # return
    return model
