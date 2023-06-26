"""Module for utility and helper functions during the object detection
and model training phase.
"""

import os
import random
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
        shuffle=True,
        collate_fn=collate_fn
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, validation_loader
