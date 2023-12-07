"""Module for utility and helper functions during the object detection
and model training phase.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Any
from xml.etree import ElementTree as et
import copy

from Detection.CustomDataset import CustomDataset


def collate_fn(batch: Any) -> Tuple:
    """
    Collate function for data loading and handling different images with varying
    classes and sizes.

    This function is used in conjunction with a PyTorch DataLoader to process a
    batch of samples. It takes a batch of samples, where each sample is a tuple
    containing an image and its associated data, and performs necessary operations
    to handle variations in image sizes and classes.

    Args:
        batch (Any):
                A batch of samples to be collated.

    Returns:
        Tuple: 
            A tuple containing the collated data. The data is organized in a way
            that handles different tensor shapes in the output, ensuring
            consistency across the batch.
    """
    return tuple(zip(*batch))


def create_dataloader(train_img_directory: str,
                      train_xml_directory: str,
                      label_mapping_dict: dict,
                      validation_img_directory: str = None,
                      validation_xml_directory: str = None,
                      train_validation_dict: dict = None,
                      train_dir_is_valid_dir: bool = False,
                      test_pattern: str = None,
                      test_list: list = [],
                      image_format: str = 'png',
                      train_transforms: Any = None,
                      validation_transforms: Any = None,
                      train_batch_size: int = 16,
                      validation_batch_size: int = 8,
                      train_split: float = 0.8) -> tuple:
    """
    Create and return data loaders for training and validation datasets.
    If training and validation directories are specified those are used.
    If the training and validation images are in the same folder and 
    random samples should be used for training and validation, specify
    ``train_dir_is_valid_dir=True`` and hand over only the training
    directory for images and XMLs.

    Args:
        train_img_directory (str):
            Path to the directory containing training images.
        train_xml_directory (str):
            Path to the directory containing training XML files.
        label_mapping_dict (dict):
            Dictionary containing the class names along with their
            corresponding integer label. XML label files often include
            the label as string but the models need integer classes.
            Example: {'class_label_1': 1, 'class_label_2': 2}
        validation_img_directory (str):
            Path to the directory containing validation images.
            Default = None
        validation_xml_directory (str): 
            Path to the directory containing validation XML files.
            Default = None
        train_dir_is_valid_dir (bool):
            Flag indicating whether the train_img_directory is the same
            as the train_xml_directory.
            Default = False
        test_pattern (str):
            Pattern for the image names that should be used as a test
            set and therefore be excluded completely during training.
            Default = None
        test_list (list):
            List of image names that should be used as a test
            set and therefore be excluded completely during training.
            Default = None
        image_format (str):
            Image file format.
            Default = 'png'
        train_transforms (Any):
            Transformations to apply to the training dataset.
            Default = None
        validation_transforms (Any):
            Transformations to apply to the validation dataset.
            Default = None
        train_batch_size (int):
            Batch size for the training data loader.
            Default = 16
        validation_batch_size (int):
            Batch size for the validation data loader.
            Default = 8
        train_split (float):
            Split ratio for training and validation data.
            Default = 0.75

    Returns:
        tuple:
            A tuple containing the training data loader and the
            validation data loader.
    """
    if train_dir_is_valid_dir:
        # get all images in directory folder
        images = [image for image in os.listdir(train_img_directory) \
                      if image.endswith(image_format)]
        # exclude images that should be used for testing
        if test_pattern:
            test_list += [image for image in images if image.startswith(test_pattern)]
        if test_list:
            images = [image for image in images if image not in test_list]
        # split in training and validation images
        if train_validation_dict:
            train_images = train_validation_dict['train_images']
            validation_images = train_validation_dict['validation_images']
        else:
            random.shuffle(images)
            train_images = images[:int(train_split*len(images))]
            validation_images = images[int(train_split*len(images)):]
        # make training dataset
        train_dataset = CustomDataset(
            image_dir=train_img_directory,
            xml_dir=train_xml_directory,
            image_list=train_images,
            image_format=image_format,
            height=512,
            width=512,
            transforms=train_transforms,
            label_mapping_dict=label_mapping_dict
        )
        # make validation dataset
        validation_dataset = CustomDataset(
            image_dir=train_img_directory,
            xml_dir=train_xml_directory,
            image_list=validation_images,
            image_format=image_format,
            height=512,
            width=512,
            transforms=validation_transforms,
            label_mapping_dict=label_mapping_dict
        )

    else:
        if (validation_img_directory is None) or \
                (validation_xml_directory is None):
            raise ValueError(f"""No validation data directory specified.""")
        # make training dataset
        train_dataset = CustomDataset(
            image_dir=train_img_directory,
            xml_dir=train_xml_directory,
            image_format=image_format,
            height=512,
            width=512,
            transforms=train_transforms,
            label_mapping_dict=label_mapping_dict
        )
        # make validation dataset
        validation_dataset = CustomDataset(
            image_dir=validation_img_directory,
            xml_dir=validation_xml_directory,
            image_format=image_format,
            height=512,
            width=512,
            transforms=validation_transforms,
            label_mapping_dict=label_mapping_dict
        )

    # make data loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True
    )
    train_val_images_dict = {'test_images': test_list,
                             'train_images': train_images,
                             'validation_images': validation_images}
    return train_loader, validation_loader, train_val_images_dict


def evaluate_loss(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> float:
    """Evaluate the average loss of a model on a given data loader.

    Args:
        model (torch.nn.Module):
            The model to evaluate.
        data_loader (torch.utils.data.DataLoader):
            The data loader containing the evaluation data.
        device (torch.device):
            The device to use for evaluation (e.g., 'cpu' or 'cuda').

    Returns:
        float:
            The average loss value.
    """
    with torch.no_grad():
        loss = []
        for data in data_loader:
            # set the model to train modus, to obtain loss
            model.train()
            # extract the data
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            # extract the losses
            loss_dict = model(images, targets)
            loss_sum = sum(loss for loss in loss_dict.values()).cpu()
            loss.append(loss_sum)
        # return average
        return np.mean(loss)
    

def train_one_epoch(model: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    device: torch.device,
                    optimizer: torch.optim.Optimizer) -> None:
    """Train the model for one epoch using the given data
    loader and optimizer.

    Args:
        model (torch.nn.Module):
            The model to train.
        data_loader (torch.utils.data.DataLoader):
            The data loader containing the training data.
        device (torch.device):
            The device to use for training (e.g., 'cpu' or 'cuda').
        optimizer (torch.optim.Optimizer):
            The optimizer to use for parameter updates.
    """
    # set model to training modus
    model.train()
    for data in data_loader:
        # set the optimizer
        optimizer.zero_grad()
        # extract the data
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        # extract the loss
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        # do the magic
        loss.backward()
        optimizer.step()


def write_out_results(output_directory: str,
                      run_name: str,
                      model: torch.nn.Module = None,
                      training_loss: list = None,
                      validation_loss: list = None,
                      training_MAP: list = None,
                      validation_MAP: list = None,
                      optimizer: torch.optim.Optimizer = None,
                      learning_rate_scheduler: torch.optim.lr_scheduler.StepLR = None,
                      train_transformations: Any = None,
                      write_out_dicts: dict = None) -> None:
    """Write out the loss, and optimizer data to
    the specified output directory.

    Args:
        output_directory (str):
            The directory to save the model and data.
        run_name (str):
            The name of the current run or experiment.
        model (torch.nn.Module, optional):
            The PyTorch model to be saved.
        training_loss (list, optional):
            The list of training loss values for each epoch.
        validation_loss (list, optional):
            The list of validation loss values for each epoch.
        training_MAP (list, optional):
            The list of training MAP values for each epoch.
        validation_MAP (list, optional):
            The list of validation MAP values for each epoch.
        optimizer (torch.optim.Optimizer, optional):
            The optimizer used for training.
        learning_rate_scheduler (torch.optim.lr_scheduler.StepLR, optional):
            The learning rate scheduler used during training.
        train_transformations (Any, optional):
            Information about the transformations applied during training.
        write_out_dicts (dict):
            A dictionary containing other dictionaries that should
            be written out. The keys are used as filenames.
    """
    # generate output directory
    save_direc = os.path.join(os.getcwd(), output_directory, run_name)
    # make directory if not exists
    if not os.path.exists(save_direc):
        os.makedirs(save_direc)
    if training_loss or validation_loss:
        loss_df = pd.DataFrame({'Epoch': range(len(training_loss)),
                                'TrainingLoss': training_loss,
                                'ValidationLoss': validation_loss})
        loss_df.to_csv(os.path.join(save_direc, 'loss_df.csv'), index=False)
        print(f"""Loss Saved ✓""")
    if training_MAP:
        training_MAP = transform_dict(training_MAP)
        training_MAP_df = pd.DataFrame(training_MAP)
        training_MAP_df.to_csv(os.path.join(save_direc, 'training_MAP.csv'),
                               index=False)
        print(f"""Training MAP Saved ✓""")
    if validation_MAP:
        validation_MAP = transform_dict(validation_MAP)
        validation_MAP_df = pd.DataFrame(validation_MAP)
        validation_MAP_df.to_csv(os.path.join(save_direc, 'validation_MAP.csv'),
                                 index=False)
        print(f"""Validation MAP Saved ✓""")
    if model:
        with open(os.path.join(save_direc, 'model.txt'), 'w+') as f:
            print(model, file=f)
        print(f"""Model Specifications Saved ✓""")
    if optimizer:
        with open(os.path.join(save_direc, 'optimizer.txt'), 'w+') as f:
            print(optimizer, file=f)
        print(f"""Optimizer Data Saved ✓""")
    if learning_rate_scheduler:
        with open(os.path.join(save_direc, 'learning_rate.txt'), 'w+') as f:
            print(learning_rate_scheduler.state_dict(), file=f)
        print(f"""Learning Rate Data Saved ✓""")
    if train_transformations:
        with open(os.path.join(save_direc, 'transformations.txt'), 'w+') as f:
            print(train_transformations, file=f)
        print(f"""Transformations Data Saved ✓""")
    if write_out_dicts:
        for key, value in write_out_dicts.items():
            with open(os.path.join(save_direc, f'{key}.json'), 'w+') as fp:
                json.dump(value, fp)
            print(f"""{key} Data Saved ✓""")


def write_out_model(model: torch.nn.Module,
                    output_directory: str,
                    run_name: str,
                    epoch: int):
    """Write out the model to the specified output directory.

    Args:
        model (torch.nn.Module):
            The model to save.
        output_directory (str):
            The directory to save the model and data.
        run_name (str):
            The name of the current run or experiment.
        epoch (int):
            The epoch iteration.
    """
    # generate output directory
    save_direc = os.path.join(os.getcwd(), output_directory, run_name)
    # make directory if not exists
    if not os.path.exists(save_direc):
        os.makedirs(save_direc)
    # make model saving directory
    model_save_direc = os.path.join(os.getcwd(), output_directory, run_name, 'models')
    if not os.path.exists(model_save_direc):
        os.makedirs(model_save_direc)
    # save model
    torch.save(model, os.path.join(model_save_direc, ('epoch_' + str(epoch) + '_model.pth')))
    print(f"""Model Saved ✓""")


def append_dicts(dict1: dict,
                 dict2: dict) -> None:
    """Appends two dictionaries and their values as lists
    when the keys are identical.

    Args:
        dict1 (dict):
            Dictionary that should be extended.
        dict2 (dict):
            Dictionary that should be appended.
    """
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], list):
                dict1[key].append(value)
            else:
                dict1[key] = [dict1[key], value]
        else:
            dict1[key] = value


def transform_dict(dict: dict) -> dict:
    """Given a dictionary where the values are list of torch tensors
    or torch tensors, the values are transformed to scalar values or
    pythonic lists.

    Args:
        dict (dict):
            Dictionary of interest.
    Returns:
        new_dict (dict):
            Dictionary with default pythonic values instead of torch
            tensors.
    """
    new_dict = {}
    # Loop through the original dictionary and extract numerical values
    for key, value in dict.items():
        # Check if the value is a list containing tensors
        if isinstance(value[0], torch.Tensor):
            new_dict[key] = [tensor.item() if (len(tensor.shape) == 0) else tensor.tolist() for tensor in value]
        else:
            # If it's not a tensor list, just assign the original value
            new_dict[key] = value
    return new_dict


def visualize_training_output(output_folder: str,
                              show_plot: bool = False,
                              save_plot: bool = True,
                              background_color: str = '#eff1f3',
                              train_color: str = '#222843',
                              validation_color: str = '#dda15e') -> None:
    """Visualize the training output including loss and accuracy plots.

    Agrs:
        output_folder (str):
            The name of the folder containing the training output files.
        show_plot (bool, optional):
            If True, display the generated plots interactively.
            Default = False
        save_plot (bool, optional):
            If True, save the generated plots in the output folder.
            Default = True
        background_color (str, optional):
            Background color for the plots.
            Default = '#eff1f3'.
        train_color (str, optional):
            Color for training-related lines in the plots.
            Default = '#222843'.
        validation_color (str, optional):
            Color for validation-related lines in the plots.
            Default = '#dda15e'.
    """
    # generate path
    path = './Output/' + output_folder + '/'

    # read in data
    loss_df = pd.read_csv(path+'loss_df.csv')
    training_map = pd.read_csv(path+'training_MAP.csv')
    validation_map = pd.read_csv(path+'validation_MAP.csv')

    # Loss Plot
    plt.figure(facecolor=background_color)
    plt.gca().set_facecolor(background_color)
    plt.plot(loss_df['TrainingLoss'], color=train_color, label='Train Loss')
    plt.plot(loss_df['ValidationLoss'], color=validation_color, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(facecolor=background_color, loc='upper left')
    if save_plot:
        plt.savefig((path+'loss_plot.png'), bbox_inches='tight', dpi=500)
    if show_plot:
        plt.show()
    plt.close()

    # Accuracy Plot
    plt.figure(facecolor=background_color)
    plt.gca().set_facecolor(background_color)
    plt.plot(training_map['map_50']*100, color=train_color, label='Train mAP50')
    plt.plot(training_map['map']*100, color=train_color, label='Train mAP50:95', 
            linestyle='dashed', alpha=0.6, linewidth=0.65)
    plt.plot(training_map['mar_100']*100, color=train_color, label='Train mAR@100',
            linestyle='dotted', alpha=0.6, linewidth=0.65)
    plt.plot(validation_map['map_50']*100, color=validation_color, label='Validation mAP50')
    plt.plot(validation_map['map']*100, color=validation_color, label='Validation mAP50:95',
            linestyle='dashed', alpha=0.6, linewidth=0.65)
    plt.plot(validation_map['mar_100']*100, color=validation_color, label='Validation mAR@100',
            linestyle='dotted', alpha=0.6, linewidth=0.65)
    plt.xlabel('Epoch')
    plt.ylabel('mAP / mAR')
    plt.ylim(0, 100)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(facecolor=background_color, loc='lower right')
    if save_plot:
        plt.savefig((path+'accuracy_plot.png'), bbox_inches='tight', dpi=500)
    if show_plot:
        plt.show()
    plt.close()

    print(f"""Visualization completed ✓""")


def extract_bboxes_from_xml(bboxes_path: str,
                            class_label_name: str = 'name') -> Tuple[list, list]:
    """
    Extract bounding boxes and labels from an XML file.

    Args:
        bboxes_path (str):
            Path to the XML file containing bounding boxes.
            Default = None.
        class_label_name (str, optional):
            Name of the class label in the XML file.
            Default = 'name'.

    Returns:
        Tuple[list, list]: Bounding boxes and labels as lists.
    """
    # initialize XML reader
    tree = et.parse(bboxes_path)
    root = tree.getroot()
    # initialize emtpy objects to store
    boxes = []
    labels = []
    # extract boxes
    for member in root.findall('object'):
        xmin = int(member.find('bndbox').find('xmin').text)
        xmax = int(member.find('bndbox').find('xmax').text)
        ymin = int(member.find('bndbox').find('ymin').text)
        ymax = int(member.find('bndbox').find('ymax').text)
        labels.append(member.find(class_label_name).text)
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes, labels

def compute_iou(box1: list,
                box2: list) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (list): List representing the first bounding box in format
            [xmin, ymin, xmax, ymax].
        box2 (list): List representing the second bounding box in format
            [xmin, ymin, xmax, ymax].

    Returns:
        float: Intersection over Union (IoU) value.
    """
    # Calculate the coordinates of the intersection rectangle
    xmin_intersection = max(box1[0], box2[0])
    ymin_intersection = max(box1[1], box2[1])
    xmax_intersection = min(box1[2], box2[2])
    ymax_intersection = min(box1[3], box2[3])

    # If there is no intersection, return IoU as 0
    if xmin_intersection >= xmax_intersection or ymin_intersection >= ymax_intersection:
        return 0.0

    # Calculate the areas of the intersection and union rectangles
    intersection_area = (xmax_intersection - xmin_intersection) * (ymax_intersection - ymin_intersection)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU value
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def get_adjusted_predictions(predictions: list,
                             threshold_dict: dict,
                             label_dict: dict = None,
                             apply_overlapping_heuristic: bool = False) -> Tuple:
    """
    Adjust predictions based on specified thresholds and overlapping heuristic.

    Args:
        predictions (list): 
            List containing model predictions with bounding boxes, labels,
            and scores.
        threshold_dict (dict):
            Dictionary mapping class labels to score thresholds.
        label_dict (dict, optional):
            Dictionary mapping integer class labels to class names.
            Default is {1: 'healthy', 2: 'infested', 3: 'dead'}.
        apply_overlapping_heuristic (bool, optional):
            Flag to apply overlapping heuristic between classes.
            Default is False.

    Returns:
        tuple:
            A tuple containing a dictionary with adjusted predictions for 
            each class and a list of tuples representing all adjusted
            bounding boxes with labels and scores.
    """
    if label_dict is None:
        label_dict = {1: 'healthy', 2: 'infested', 3: 'dead'}
    # extract boxes and labels from prediction
    boxes = predictions[0]['boxes'].numpy()
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()
    # prepare loop helper
    res_dict = {}
    for _, v in label_dict.items():
        res_dict[v] = {'boxes': [], 'scores': [], 'labels': []}
    # loop through predictions and extract adjusted once
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold_dict[label_dict[label]]:
            res_dict[label_dict[label]]['boxes'].append(box.tolist())
            res_dict[label_dict[label]]['scores'].append(round(score, 4))
            res_dict[label_dict[label]]['labels'].append(label)
    # remove boxes that overlap by threshold
    for k, v in label_dict.items():
        b, s, l = filter_boxes(res_dict[v]['boxes'],
                               res_dict[v]['scores'],
                               k,
                               0.25)
        res_dict[v]['boxes'] = b
        res_dict[v]['scores'] = s
        res_dict[v]['labels'] = l
    # apply heuristic for overlapping boxes between classes
    if apply_overlapping_heuristic:
        dict_healthy = copy.deepcopy(res_dict['healthy'])
        dict_infested = copy.deepcopy(res_dict['infested'])
        dict_dead = copy.deepcopy(res_dict['dead'])
        dict_infested = pop_overlapping_boxes(dict_infested, dict_healthy, score_threshold=0.4)
        dict_infested = pop_overlapping_boxes(dict_infested, dict_dead, score_threshold=0.7)
        res_dict = {
            'healthy' : dict_healthy,
            'infested': dict_infested,
            'dead': dict_dead
        }
    # get an everything list again
    b = []; s = []; l = []
    for _, v in label_dict.items():
        b = b + res_dict[v]['boxes']
        l = l + res_dict[v]['labels']
        s = s + res_dict[v]['scores']
    all_boxes = list(zip(b, l, s))
    return res_dict, all_boxes


def get_adjusted_ground_truth(ground_truth: list,
                              rev_label_dict: dict = None) -> dict:
    """
    Adjust ground truth based on class labels.

    Args:
        ground_truth (list): 
            List containing ground truth information with bounding boxes and labels.
        rev_label_dict (dict, optional):
            Dictionary mapping class names to integer labels.
            Default is {'healthy': 1, 'infested': 2, 'dead': 3}.

    Returns:
        dict:
            A dictionary containing adjusted ground truth information for each class.
    """
    if rev_label_dict is None:
        rev_label_dict = {'healthy': 1,'infested': 2, 'dead': 3}
    # prepare loop helper
    gt_dict = {}
    for k, _ in rev_label_dict.items():
        gt_dict[k] = {'boxes': [], 'labels': []}
    # loop through true labels and extract based on class
    for b in ground_truth:
        box = b[0]
        label = b[1]
        gt_dict[label]['boxes'].append(box)
        gt_dict[label]['labels'].append(rev_label_dict[label])
    return gt_dict


def precision_recall_f1score_detection(pred_boxes: list,
                                       true_boxes: list,
                                       iou_threshold: float) -> dict:
    """
    Calculate precision, recall, and F1 score for object detection.

    Args:
        pred_boxes (list): 
            List of predicted bounding boxes.
        true_boxes (list): 
            List of true bounding boxes.
        iou_threshold (float): 
            Intersection over Union (IoU) threshold for matching predicted
            and true boxes.

    Returns:
        dict:
            A dictionary containing the following metrics:
                - 'tp' (int): True positives.
                - 'fp' (int): False positives.
                - 'fn' (int): False negatives.
                - 'precision' (float): Precision score.
                - 'recall' (float): Recall score.
                - 'f1_score' (float): F1 score.
    """
    tb = copy.deepcopy(true_boxes)
    tp = 0
    fp = 0
    fn = 0
    # loop prediction boxes
    for pred_box in pred_boxes:
        iou_max = 0
        match_index = -1
        # check with true box
        for i, true_box in enumerate(tb):
            iou = compute_iou(pred_box, true_box)
            if iou > iou_max:
                iou_max = iou
                match_index = i
        if iou_max >= iou_threshold:
            tp += 1
            tb.pop(match_index)
        else:
            fp += 1

    fn = len(tb)

    # compute precision/recall scores
    precision = (tp + fp) and tp / (tp + fp) or 0
    recall = (tp + fn) and tp / (tp + fn) or 0
    f1_score = (tp + fp + fn) and 2*tp / (2*tp + fp + fn) or 0

    result_dict = {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1_score': f1_score
    }
    return result_dict


def filter_boxes(pred_boxes: list,
                 scores: list,
                 label: int,
                 iou_threshold: float) -> Tuple:
    """
    Filter overlapping boxes based on the specified IoU threshold.

    Args:
        pred_boxes (list): 
            List of predicted bounding boxes.
        scores (list): 
            List of corresponding confidence scores for the predicted boxes.
        label (int): 
            Class label for the boxes.
        iou_threshold (float): 
            Intersection over Union (IoU) threshold for filtering overlapping boxes.

    Returns:
        tuple:
            A tuple containing lists of filtered bounding boxes, corresponding scores,
            and labels.
    """
    # Combine predicted boxes and scores into a list of tuples
    box_with_scores = list(zip(pred_boxes, scores))

    # Sort the list by scores in descending order
    box_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Initialize a list to store the filtered boxes
    filtered_boxes = []
    filtered_scores = []
    labels = []

    while box_with_scores:
        box1, score1 = box_with_scores.pop(0)
        boxes_to_remove = []

        for i, (box2, score2) in enumerate(box_with_scores):
            iou = compute_iou(box1, box2)
            x = box_within(box1, box2)
            y = box_within(box2, box1)

            if (iou >= iou_threshold) or x or y:
                # Remove the box with the lower score
                if score1 > score2:
                    boxes_to_remove.append(i)
                else:
                    boxes_to_remove.append(0)

        # Remove the marked boxes
        box_with_scores = [box_with_scores[i] for i in range(len(box_with_scores)) if i not in boxes_to_remove]

        # Add the box with the higher score to the filtered list
        filtered_boxes.append(box1)
        filtered_scores.append(score1)
        labels.append(label)

    return filtered_boxes, filtered_scores, labels


def plot_pred_vs_true(image_path: str,
                      all_boxes_pred: list,
                      all_boxes_ground_truth: list,
                      label_dict: dict = None,
                      color_dict_1: dict = None,
                      color_dict_2: dict = None,
                      show: bool = True,
                      save_path: str = None) -> None:
    """
    Plot side-by-side visualizations of predicted and ground truth bounding
    boxes on an image.

    Args:
        image_path (str):
            Path to the image file.
        all_boxes_pred (list):
            List of tuples containing predicted bounding boxes, labels,
            and scores.
        all_boxes_ground_truth (list):
            List of tuples containing ground truth bounding boxes and labels.
        label_dict (dict, optional):
            Dictionary mapping class labels to class names.
            Default is {1: 'healthy', 2: 'infested', 3: 'dead'}.
        color_dict_1 (dict, optional):
            Dictionary mapping class labels to colors for predicted boxes.
            Default is {1: '#ffffff', 2: '#ffa500', 3: '#cb577a'}.
        color_dict_2 (dict, optional):
            Dictionary mapping class names to colors for ground truth boxes.
            Default is {'healthy': '#ffffff', 'infested': '#ffa500', 'dead': '#cb577a'}.
        show (bool, optional):
            Flag indicating whether to display the plot. Default is True.
        save_path (str, optional):
            If specified, the plot will be saved to this file path.
    """
    if label_dict is None:
        label_dict = {1: 'healthy', 2: 'infested', 3: 'dead'}
    if color_dict_1 is None:
        color_dict_1 = {1: '#ffffff', 2: '#ffa500', 3: '#cb577a'}
    if color_dict_2 is None:
        color_dict_2 = {'healthy': '#ffffff', 'infested': '#ffa500', 'dead': '#cb577a'}

    # read in image
    image = plt.imread(image_path)

    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # plot predictions
    for box, label, score in all_boxes_pred:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                    linewidth=1.5,
                                    edgecolor=color_dict_1[label],
                                    facecolor='none')
        ax[0].add_patch(rect)
        ax[0].text(box[0], box[1]-2,
                f'{label_dict[label]} - {score:.2f}',
                fontsize=11, color=color_dict_1[label],
                fontweight='bold')
    ax[0].imshow(image)
    ax[0].axis('off')

    # plot ground truth
    for box, label in all_boxes_ground_truth:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                    linewidth=1.5,
                                    edgecolor=color_dict_2[label],
                                    facecolor='none')
        ax[1].add_patch(rect)
        ax[1].text(box[0], box[1]-2,
                f'{label}',
                fontsize=11, color=color_dict_2[label],
                fontweight='bold')
    ax[1].imshow(image)
    ax[1].axis('off')

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    if not show:
        plt.close(fig)


def box_within(box1: Tuple,
               box2: Tuple) -> bool:
    """
    Check if the first bounding box is completely contained within the second bounding box.

    Args:
        box1 (tuple):
            Coordinates (x_min, y_min, x_max, y_max) of the first bounding box.
        box2 (tuple):
            Coordinates (x_min, y_min, x_max, y_max) of the second bounding box.

    Returns:
        bool:
            True if box1 is completely contained within box2, False otherwise.
    """
    return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]


def pop_overlapping_boxes(dict_infested: dict,
                          dict_other: dict,
                          score_threshold: float) -> dict:
    """
    Remove overlapping boxes in dict_infested based on overlap with dict_other.

    Args:
        dict_infested (dict):
            Dictionary containing 'boxes', 'scores', and 'labels' lists for infested class.
        dict_other (dict):
            Dictionary containing 'boxes', 'scores', and 'labels' lists for another class.
        score_threshold (float):
            Score threshold for retaining infested boxes.

    Returns:
        dict:
            Dictionary with non-overlapping infested boxes based on the given score threshold.
    """
    new_dict_infested = {'boxes': [], 'scores': [], 'labels': []}
    
    for box_inf, score_inf, label_inf in list(zip(dict_infested['boxes'], dict_infested['scores'], dict_infested['labels'])):
        keep_flag = True
        for box, score, label in list(zip(dict_other['boxes'], dict_other['scores'], dict_other['labels'])):

            overlap = ((compute_iou(box_inf, box) >= 0.5) or
                        box_within(box_inf, box) or
                        box_within(box, box_inf))
            score = score_inf / score

            if (overlap and (score < score_threshold)):
                keep_flag = False

        if keep_flag:
            new_dict_infested['boxes'].append(box_inf)
            new_dict_infested['scores'].append(score_inf)
            new_dict_infested['labels'].append(label_inf)

    return new_dict_infested

