"""
"""

import torch
from torchmetrics.detection import MeanAveragePrecision


def evaluate_MAP(model, dataloader, device):
    """
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
