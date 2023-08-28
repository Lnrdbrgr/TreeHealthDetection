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
    all_pixels = mask.shape[1] * mask.shape[1]
    acc = correct_pixels / all_pixels
    return acc


def jaccard_accuracy(pred: torch.tensor, mask) -> list:
    """Returns accuracy for class 0, 1, 2, ...
    """
    if not pred.shape == mask.shape:
        pred = torch.argmax(pred, dim=1)
    pred = pred.flatten().numpy()
    mask = mask.flatten().numpy()
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
        jaccard_acc = []
        for images, target in data_loader:
            # move data to device
            images = images.to(device)
            target = target.to(device)
            # make prediction
            preds = model(images)
            # store metrics
            loss.append(loss_fn(preds, target))
            pixel_acc.append(pixel_accuracy(pred=preds, mask=target))
            jaccard_acc.append(jaccard_accuracy(pred=preds, mask=target))
    loss = np.mean(loss)
    pixel_acc = np.mean(pixel_acc)
    jaccard_acc = np.mean(np.array(jaccard_acc), axis=0)
    return loss, pixel_acc, jaccard_acc


def write_out_results(output_directory: str,
                      run_name: str,
                      training_loss: list = None,
                      validation_loss: list = None,
                      training_pixel_acc: list = None,
                      validation_pixel_acc: list = None,
                      training_jaccard: list = None,
                      validation_jaccard: list = None,
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
    if training_jaccard:
        training_jaccard_df = pd.DataFrame(
            training_jaccard,
            columns=['Background', 'Healthy', 'Infested', 'Dead']
        )
        training_jaccard_df['Epoch'] = range(training_jaccard_df.shape[0])
        training_jaccard_df.to_csv(os.path.join(save_direc, 'train_jaccard.csv'),
                                   index=False)
        print(f"""Jaccard Accuracy Saved ✓""")
    if validation_jaccard:
        validation_jaccard_df = pd.DataFrame(
            validation_jaccard,
            columns=['Background', 'Healthy', 'Infested', 'Dead']
        )
        validation_jaccard_df['Epoch'] = range(validation_jaccard_df.shape[0])
        validation_jaccard_df.to_csv(os.path.join(save_direc, 'validation_jaccard.csv'),
                                   index=False)
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
