"""Module for utility and helper functions during the segmentation prediction
and model training phase.
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Any
from sklearn.metrics import jaccard_score

from Segmentation.Custom3DSegmentationDataset import Custom3DSegmentationDataset


def create_dataloader(data_directory: str,
                      train_images: list | str = None,
                      validation_images: list | str = None,
                      test_images: list | str = None,
                      random_train_val: bool = False,
                      train_val_split: float = 0.8,
                      resize_to: Tuple = (512, 512),
                      train_transforms: Any = None,
                      validation_transforms: Any = None) -> Tuple:
    """
    Create PyTorch data loaders for training and validation from a directory of
    3D segmentation data.

    Args:
        data_directory (str):
            Path to the root directory of the dataset.
        train_images (list or str, optional):
            List of specific data folders to include in the training set. If not
            provided, the training set will be generated based on 'random_train_val'
            and 'train_val_split'.
            Default is None.
        validation_images (list or str, optional):
            List of specific data folders to include in the validation set. If
            not provided, the validation set will be generated based on
            'random_train_val' and 'train_val_split'.
            Default is None.
        test_images (list or str, optional):
            List of specific data folders to include in the test set.
            Default is None.
        random_train_val (bool, optional):
            If True, randomly split the data into training and validation sets
            based on 'train_val_split'. If False, use the provided 'train_images'
            and 'validation_images' lists for training and validation.
            Default is False.
        train_val_split (float, optional):
            Fraction of the data to be used for training if 'random_train_val'
            is True.
            Default is 0.8.
        resize_to (Tuple, optional):
            Tuple specifying the target size for resizing images in the format
            (height, width).
            Default is (512, 512).
        train_transforms (Any, optional):
            Augmentation transforms to be applied to the training set.
            Default is None.
        validation_transforms (Any, optional):
            Augmentation transforms to be applied to the validation set.
            Default is None.

    Returns:
        Tuple[DataLoader, DataLoader]:
            Tuple containing the PyTorch data loaders for training and
            validation sets.
    """
    if random_train_val and (train_images or validation_images):
        raise ValueError(f"""Cannot specify values and random split.""")
    if not random_train_val and not train_images and not validation_images:
        raise ValueError(f"""Specify how train and validation sets are set.""")
    all_images = os.listdir(data_directory)
    if isinstance(test_images, str):
        test_images = [i for i in all_images if test_images in i]
    if test_images is not None:
        all_images = [i for i in all_images if i not in test_images]
    if isinstance(train_images, str):
        train_images = [i for i in all_images if train_images in i]
    if random_train_val:
        train_images = random.sample(all_images, int(train_val_split*len(all_images)))
    if isinstance(validation_images, str):
        validation_images = [i for i in all_images if validation_images in i]
        train_images = list(set(all_images) - set(validation_images))
    else:
        validation_images = list(set(all_images) - set(train_images))
    if not train_images and not validation_images:
        raise ValueError(f"""No training or validation lists generated.""")
    
    # make custom datasets
    train_dataset = Custom3DSegmentationDataset(
        data_dir=data_directory,
        data_list=train_images,
        resize_to=resize_to,
        transforms=train_transforms
    )
    validation_dataset = Custom3DSegmentationDataset(
        data_dir=data_directory,
        data_list=validation_images,
        resize_to=resize_to,
        transforms=validation_transforms
    )

    # make dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_loader, validation_loader


def pixel_accuracy(pred: torch.Tensor,
                   mask: torch.Tensor) -> float:
    """
    Compute pixel accuracy between predicted segmentation and ground truth mask.

    Args:
        pred (torch.Tensor):
            Predicted segmentation map with shape (batch_size, num_classes,
            height, width). If the predictions are one-hot encoded, the
            function will use argmax to obtain class labels.
        mask (torch.Tensor):
            Ground truth mask with shape (batch_size, height, width).

    Returns:
        float:
            Pixel accuracy, defined as the ratio of correctly classified
            pixels to the total number of pixels.
    """
    if not pred.shape == mask.shape:
        pred = torch.argmax(pred, dim=1)
    correct_pixels = (pred == mask).sum()
    all_pixels = mask.shape[1] * mask.shape[2]
    acc = correct_pixels / all_pixels
    return acc


def evaluate_segmentation_accuracy(model: torch.nn.Module,
                                   data_loader: torch.utils.data.DataLoader,
                                   device: torch.device,
                                   loss_fn: torch.nn.modules.loss) -> Tuple:
    """
    Evaluate segmentation accuracy of a model on a given data loader.

    Args:
        model (torch.nn.Module):
            Segmentation model to evaluate.
        data_loader (torch.utils.data.DataLoader):
            DataLoader providing batches of data for evaluation.
        device (torch.device):
            Device on which to perform the evaluation (e.g., 'cuda' or 'cpu').
        loss_fn (torch.nn.modules.loss):
            Loss function to compute during evaluation.

    Returns:
        Tuple[float, float, pd.DataFrame]:
            Tuple containing the average loss, pixel accuracy, and a DataFrame
            with class-wise metrics (precision, recall, and F1-score).
    """
    with torch.no_grad():
        # move model to device
        model = model.to(device)
        # initiate loop objects
        loss = []
        pixel_acc = []
        metrics_df = pd.DataFrame()
        for images, target in data_loader:
            # move data to device
            images = images.to(device)
            target = target.to(device)
            # make prediction
            preds = model(images)
            # store metrics
            loss.append(loss_fn(preds, target).cpu())
            pixel_acc.append(pixel_accuracy(pred=preds, mask=target).cpu())
            # store F1-Score
            class_predictions = torch.argmax(preds, dim=1)
            metrics_df = pd.concat(
                [metrics_df, precision_recall_f1score(true_mask=target.cpu().numpy(), pred_mask=class_predictions.cpu().numpy())]
            )
    loss = np.mean(loss)
    pixel_acc = np.mean(pixel_acc)
    metrics_df = metrics_df.groupby('metric').mean().reset_index()
    return loss, pixel_acc, metrics_df


def write_out_results(output_directory: str,
                      run_name: str,
                      training_loss: list = None,
                      validation_loss: list = None,
                      training_pixel_acc: list = None,
                      validation_pixel_acc: list = None,
                      training_metrics_df: pd.DataFrame = None,
                      validation_metrics_df: pd.DataFrame = None,
                      optimizer: torch.optim.Optimizer = None,
                      learning_rate_scheduler: torch.optim.lr_scheduler.StepLR = None,
                      model: torch.nn.Module = None) -> None:
    """
    Write out training and validation results, metrics, and model-related
    information.

    Args:
        output_directory (str):
            Directory where results will be saved.
        run_name (str):
            Name of the run or experiment.
        training_loss (list, optional):
            List containing training loss values for each epoch.
        validation_loss (list, optional):
            List containing validation loss values for each epoch.
        training_pixel_acc (list, optional):
            List containing training pixel accuracy values for each epoch.
        validation_pixel_acc (list, optional):
            List containing validation pixel accuracy values for each epoch.
        training_metrics_df (pd.DataFrame, optional):
            DataFrame containing training metrics (precision, recall, F1-score)
            for each class.
        validation_metrics_df (pd.DataFrame, optional):
            DataFrame containing validation metrics (precision, recall, F1-score)
            for each class.
        optimizer (torch.optim.Optimizer, optional):
            Optimizer used for training the model.
        learning_rate_scheduler (torch.optim.lr_scheduler.StepLR, optional):
            Learning rate scheduler used during training.
        model (torch.nn.Module, optional):
            Trained segmentation model.

    Returns:
        None: Results are saved to the specified output directory.
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
    if training_pixel_acc or validation_pixel_acc:
        loss_df = pd.DataFrame({'Epoch': range(len(training_pixel_acc)),
                                'TrainingAccuracy': training_pixel_acc,
                                'ValidationAccuracy': validation_pixel_acc})
        loss_df.to_csv(os.path.join(save_direc, 'pixel_accuracy.csv'), index=False)
        print(f"""Pixel Accuracy Saved ✓""")
    if training_metrics_df is not None:
        training_metrics_df.to_csv(os.path.join(save_direc, 'training_metrics.csv'), index=False)
    if validation_metrics_df is not None:
        validation_metrics_df.to_csv(os.path.join(save_direc, 'validation_metrics.csv'), index=False)
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


def write_out_model(model: torch.nn.Module,
                    output_directory: str,
                    run_name: str,
                    epoch: int = None):
    """Write out the model to the specified output directory.

    Args:
        model (torch.nn.Module):
            The model to save.
        output_directory (str):
            The directory to save the model and data.
        run_name (str):
            The name of the current run or experiment.
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


def precision_recall_f1score(true_mask: np.array,
                             pred_mask: np.array,
                             class_translation_dict: dict = {0: 'background', 1: 'healthy', 2: 'infested', 3: 'dead'}) -> pd.DataFrame:
    """
    Calculate precision, recall, and F1-score for each class in a segmentation task.

    Args:
        true_mask (np.array):
            Ground truth segmentation mask.
        pred_mask (np.array):
            Predicted segmentation mask.
        class_translation_dict (dict, optional):
            Dictionary to translate class labels to corresponding names.
            Defaults to {0: 'background', 1: 'healthy', 2: 'infested', 3: 'dead'}.

    Returns:
        pd.DataFrame:
            DataFrame containing precision, recall, and F1-score for each class,
            along with other metrics.
    """
    classes = np.sort(np.unique(true_mask))
    result_dict = {'metric': ['no_pixels', 'share_pixels', 'precision', 'recall', 'f1_score']}
    for c in classes:
        tp = np.logical_and(true_mask == c, pred_mask == c).sum()
        fp = np.logical_and(true_mask != c, pred_mask == c).sum()
        fn = np.logical_and(true_mask == c, pred_mask != c).sum()
        # conditions account for division by zero
        no_pixels = (true_mask == c).sum()
        share_pixels = no_pixels / true_mask.size
        precision = (tp + fp) and tp / (tp + fp) or 0
        recall = (tp + fn) and tp / (tp + fn) or 0
        f1_score = 2*tp / (2*tp + fp + fn)
        result_dict[class_translation_dict[c]] = [no_pixels, share_pixels, precision, recall, f1_score]
    result_df = pd.DataFrame(result_dict)
    return result_df
