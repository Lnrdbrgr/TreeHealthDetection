"""Module for utility and helper functions for object detection
accuracy evaluation.
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision


def evaluate_MAP(model: torch.nn.Module,
                 dataloader: DataLoader,
                 device: torch.device) -> dict:
    """Evaluate the Mean Average Precision (MAP) of an
    object detection model..

    Args:
        model (torch.nn.Module):
            The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader):
            The DataLoader providing the dataset for evaluation.
        device (torch.device):
            The device to which the model and data should be moved
            (e.g., 'cuda' for GPU or 'cpu' for CPU).
    Returns:
        map (dict):
            The Mean Average Precision (MAP) score in a dictionary
            for multiple specifications.

    """
    MAP_class = MeanAveragePrecision()
    with torch.no_grad():
        model.eval()
        for images, target in dataloader:
            images = list(image.to(device) for image in images)
            predictions = model(images)
            predictions = [{k: v.to(torch.device("cpu")) for k, v in \
                            t.items()} for t in predictions]
            MAP_class.update(preds=predictions, target=list(target))
        map = MAP_class.compute()
    return map
