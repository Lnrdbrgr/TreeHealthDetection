"""
"""

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch


def evaluate_coco_MAP(model, dataloader, device):
    """
    """
    # set model to evaluation mode
    model.eval()
    # extract ground truth data
    coco_api_dataset = convert_to_coco_api(dataloader.dataset)
    coco_eval = COCOeval(cocoGt=coco_api_dataset, iouType='bbox')
    eval_stats = []

    # loop through validation set and compute MAP
    for data in dataloader:
        # extract the data
        images, targets = data
        images = list(image.to(device) for image in images)
        # make bbox predictions
        predictions = model(images)
        # move to cpu
        predictions = [{k: v.to(torch.device('cpu')) for k, v in pred.items()} for pred in predictions]
        targets = [{k: v.to(torch.device('cpu')) for k, v in target.items()} for target in targets]
        # pair with image id
        predictions = {target['image_id'].item(): pred for target, pred in zip(targets, predictions)}
        # make coco compatible
        results = prepare_for_coco_detection(predictions)
        # add to coco evaluation class
        coco_dt = COCO.loadRes(coco_api_dataset, results)
        coco_eval.cocoDt = coco_dt
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        eval_stats.append(coco_eval.stats)

    # return average MAP over all images
    mean_eval_stats = np.array(eval_stats).mean(axis=0)
    return mean_eval_stats


def convert_to_coco_api(ds):
    """Not mine, credit to CocoAPI
    https://github.com/cocodataset/cocoapi/tree/master
    """
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def prepare_for_coco_detection(predictions):
    """Not mine, credit to CocoAPI
    """
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def convert_to_xywh(boxes):
    """Not mine, credit to CocoAPI
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
