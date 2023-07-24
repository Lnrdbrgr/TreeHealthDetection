"""Module for utility and helper functions during the object detection
and model training phase.
"""

import matplotlib.pyplot as plt
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


def write_out_results(output_directory: str,
                      run_name: str,
                      epoch: int = None,
                      training_loss: list = None,
                      validation_loss: list = None,
                      training_MAP: list = None,
                      validation_MAP: list = None,
                      optimizer: torch.optim.Optimizer = None,
                      learning_rate_scheduler: torch.optim.lr_scheduler.StepLR = None,
                      train_transformations: Any = None) -> None:
    """Write out the loss, and optimizer data to
    the specified output directory.

    Args:
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
    if learning_rate_scheduler:
        with open(os.path.join(save_direc, 'learning_rate.txt'), 'w+') as f:
            print(learning_rate_scheduler.state_dict(), file=f)
        print(f"""Learning Rate Data Saved ✓""")
    if train_transformations:
        with open(os.path.join(save_direc, 'transformations.txt'), 'w+') as f:
            print(train_transformations, file=f)
        print(f"""Transformations Data Saved ✓""")


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
    path = '../Detection/Output/' + output_folder + '/'

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

    # Accuracy Plot
    plt.figure(facecolor=background_color)
    plt.gca().set_facecolor(background_color)
    plt.plot(training_map['map']*100, color=train_color, label='Train mAP')
    plt.plot(training_map['map_50']*100, color=train_color, label='Train mAP50', 
            linestyle='dashed', alpha=0.6, linewidth=0.65)
    plt.plot(training_map['mar_100']*100, color=train_color, label='Train mAR@100',
            linestyle='dotted', alpha=0.6, linewidth=0.65)
    plt.plot(validation_map['map']*100, color=validation_color, label='Validation mAP')
    plt.plot(validation_map['map_50']*100, color=validation_color, label='Validation mAP50',
            linestyle='dashed', alpha=0.6, linewidth=0.65)
    plt.plot(validation_map['mar_100']*100, color=validation_color, label='Validation mAR@100',
            linestyle='dotted', alpha=0.6, linewidth=0.65)
    plt.xlabel('Epoch')
    plt.ylabel('mAP / mAR (in %)')
    plt.ylim(0, 100)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(facecolor=background_color, loc='upper left')
    if save_plot:
        plt.savefig((path+'accuracy_plot.png'), bbox_inches='tight', dpi=500)
    if show_plot:
        plt.show()

    print(f"""Visualization completed ✓""")
