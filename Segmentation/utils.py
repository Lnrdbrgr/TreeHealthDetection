"""
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
                      validation_transforms: Any = None):
    """
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


def pixel_accuracy(pred, mask):
    """
    """
    if not pred.shape == mask.shape:
        pred = torch.argmax(pred, dim=1)
    correct_pixels = (pred == mask).sum()
    all_pixels = mask.shape[1] * mask.shape[2]
    acc = correct_pixels / all_pixels
    return acc


def jaccard_accuracy(pred: torch.tensor, mask):
    """Returns accuracy for class 0, 1, 2, ...
    """ 
    if not pred.shape == mask.shape:
        pred = torch.argmax(pred, dim=1)
    pred = pred.cpu().flatten().numpy()
    mask = mask.cpu().flatten().numpy()
    jaccard = jaccard_score(pred, mask, average=None)
    return jaccard


def evaluate_segmentation_accuracy(model: torch.nn.Module,
                                   data_loader: torch.utils.data.DataLoader,
                                   device: torch.device,
                                   loss_fn: torch.nn.modules.loss) -> Tuple:
    """
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
                [metrics_df, precision_recall_f1score(true_mask=target.numpy(), pred_mask=class_predictions.numpy())]
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
