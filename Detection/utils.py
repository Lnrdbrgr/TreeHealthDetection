"""Module for utility and helper functions during the object detection
and model training phase.
"""

import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Any
from xml.etree import ElementTree as et

from CustomDataset import CustomDataset


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
                      validation_img_directory: str = None,
                      validation_xml_directory: str = None,
                      train_dir_is_valid_dir: bool = False,
                      image_format: str = 'png',
                      train_transforms: Any = None,
                      validation_transforms: Any = None,
                      train_batch_size: int = 16,
                      validation_batch_size: int = 8,
                      train_split: float = 0.75) -> tuple:
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
        # split in training and validation images
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
            transforms=train_transforms
        )
        # make validation dataset
        validation_dataset = CustomDataset(
            image_dir=train_img_directory,
            xml_dir=train_xml_directory,
            image_list=validation_images,
            image_format=image_format,
            height=512,
            width=512,
            transforms=validation_transforms
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
            transforms=train_transforms
        )
        # make validation dataset
        validation_dataset = CustomDataset(
            image_dir=validation_img_directory,
            xml_dir=validation_xml_directory,
            image_format=image_format,
            height=512,
            width=512,
            transforms=validation_transforms
        )

    # make data loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, validation_loader


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


def write_out_results(model: torch.nn.Module,
                      output_directory: str,
                      run_name: str,
                      epoch: int = None,
                      training_loss: list = None,
                      validation_loss: list = None,
                      training_MAP: list = None,
                      validation_MAP: list = None,
                      optimizer: torch.optim.Optimizer = None,
                      train_transformations = None) -> None:
    """Write out the model, loss, and optimizer data to
    the specified output directory.

    Args:
        model (torch.nn.Module):
            The model to save.
        output_directory (str):
            The directory to save the model and data.
        run_name (str):
            The name of the current run or experiment.
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
    if optimizer:
        with open(os.path.join(save_direc, 'optimizer.txt'), 'w+') as f:
            print(optimizer, file=f)
        print(f"""Optimizer Data Saved ✓""")
    if train_transformations:
        with open(os.path.join(save_direc, 'transformations.txt'), 'w+') as f:
            print(train_transformations, file=f)
        print(f"""Transformations Data Saved ✓""")


def append_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], list):
                dict1[key].append(value)
            else:
                dict1[key] = [dict1[key], value]
        else:
            dict1[key] = value


def transform_dict(dict):
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
