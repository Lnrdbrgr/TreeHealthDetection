"""
"""

import albumentations as A
import torch
import cv2
import numpy as np
import os
import glob as glob
from torch.utils.data import Dataset
from typing import Any, Tuple
import matplotlib.pyplot as plt

import pydantic
from xml.etree import ElementTree as et


class Custom3DSegmentationDataset(Dataset, pydantic.BaseModel):
    """
    """
    data_dir: str
    data_list: list | None = None
    resize_to: Tuple
    transforms: Any = None

    @pydantic.root_validator(skip_on_failure=True)
    @classmethod
    def extract_all_images(cls, values: dict) -> dict:
        """
        """
        data_dir = values.get('data_dir')
        data_list = values.get('data_list')
        if data_list is not None and len(data_list) != 0:
            values['data_folder'] = data_list
        else:
            # if no data list is provided use all data in directory
            data_folder = os.listdir(data_dir)
            values['data_folder'] = data_folder
        # add resize transformer
        resize_to = values.get('resize_to')
        values['resize_transform'] = A.Compose([
            A.Resize(resize_to[0], resize_to[1], p=1,
                     interpolation=cv2.INTER_NEAREST)
        ])
        return values
        
    def __getitem__(self, index: int) -> tuple[torch.tensor, dict]:
        """
        Get an item from the dataset.
        Method necessary for torch datasets, used to extract data in 
        batch iterations.
        
        Args:
            index (int):
                Index of the item.

        Returns:
            tuple[torch.tensor, dict]:
                Tuple containing the resized image and target dictionary.
        """
        # get folder with images
        folder = self.data_folder[index]
        path = self.data_dir + folder
        # read in mask
        mask = plt.imread(path + '/mask.png')
        mask *= 255
        mask = self.resize_transform(image=mask)['image']
        # transform if applicable
        if self.transforms:
            transform_mask = self.transforms(image=mask)
            mask = transform_mask['image']
        mask = torch.tensor(mask).long()
        # extract images
        images = os.listdir(path)
        images = [i for i in images if 'image' in i]
        images = sorted(images)
        # read in every image
        image_list = []
        for i in images:
            image = cv2.imread(path + '/' + i)
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # resize
            image = self.resize_transform(image=image)['image']
            # transform if applicable
            if self.transforms:
                image = A.ReplayCompose.replay(transform_mask['replay'], image=image)
                image = image['image']
            # standardize
            image /= 255.0
            image_list.append(image)
        out_images = torch.tensor(np.array([image_list]))
        out_images = out_images.permute([0, 4, 1, 2, 3])
        out_images = out_images.squeeze(0)
        

        return out_images, mask
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_folder)
