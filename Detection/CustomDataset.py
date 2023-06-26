"""This Module hold the CustomDataset Class which inherits from pytorch
Dataset for accessing images along with their meta information for
training or evaluation.
"""

import torch
import cv2
import numpy as np
import os
import glob as glob
from torch.utils.data import Dataset
from typing import Any, Tuple

import pydantic
from xml.etree import ElementTree as et


class CustomDataset(Dataset, pydantic.BaseModel):
    """
    Custom dataset that inherits from pytorch Dataset for accessing
    images along with their meta information for training or evaluation.

    Args:
        image_dir (str):
            Directory path containing the images.
        image_list (list[str] | None):
            List containing the image names to be used in this dataset
            when not all images should be used.
            Default = None
        xml_dir (str):
            Directory path containing the XML files with annotations.
        image_format str:
            Format of the images.
        height (int):
            Height to resize the images.
        width (int):
            Width to resize the images.
        transforms (Any):
            Image transformation function.
            Default = None.
    """
    image_dir: str
    image_list: list | None = None
    xml_dir: str
    image_format: str
    height: int
    width: int
    transforms: Any = None

    @pydantic.root_validator
    def extract_all_images(cls, values: dict) -> dict:
        """
        Root validator to extract all image paths. If an image list is
        given then these images will be used. If no image list is given
        then all images in the image directory are used.
        Pydantic Root validators are executed every time on initiation
        of the class and not to be called from outside the class.

        Args:
            value (dict):
                Dictionary of attribute values.

        Returns:
            values (dict):
                Updated attribute values.
        """
        image_dir = values.get('image_dir')
        image_list = values.get('image_list')
        image_format = values.get('image_format')
        if image_list is not None and len(image_list) != 0:
            if len(image_list) < 5:
                print(f"""Warning\n Length of specified images is really
                      small, check that batch size in dataloader corresponds
                      to number of images in dataset.""")
            # if image list is provided return as images
            values['images'] = sorted(image_list)
            return values
        else:
            # if no image list is provided use all images in directory
            images = [image for image in os.listdir(image_dir) \
                      if image.endswith(image_format)]
            values['images'] = sorted(images)
            return values
        
    def __getitem__(self, index: int) -> tuple[torch.tensor, dict]:
        """
        Get an item from the dataset.
        Method necessary for torch datasets, used to extract data in 
        batch iterations.
        
        Args
            index (int):
                Index of the item.

        Returns:
            tuple[torch.tensor, dict]:
                Tuple containing the resized image and target dictionary.
        """
        # get image name and path
        image_name = self.images[index]
        image_path = self.image_dir + '/' + image_name
        # read the image
        image = cv2.imread(image_path)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # extract original shape
        image_height = image.shape[0]
        image_width = image.shape[1]
        # resize to uniform height and width
        image_resized = cv2.resize(image, (self.width, self.height))
        # standardize
        image_resized /= 255.0

        # get box coordinates and labels
        annotations_file_name = '.'.join(image_name.split('.')[:-1]) + '.xml'
        annotations_file_path = self.xml_dir + '/' + annotations_file_name
        boxes, labels = self.extract_bboxes_from_xml(bboxes_path=annotations_file_path,
                                                     class_label_name='class_no')
        # resize bounding boxes to new measures
        boxes_resized = []
        for box in boxes:
            boxes_resized.append(
                self.resize_boxes(box, image_height, image_width,
                                  self.height, self.width)
            )
        # move to pytorch tensor for further processing
        boxes = torch.as_tensor(boxes_resized, dtype=torch.float32)
        # get area of boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # initiate crowd instance
        # (for overlapping objects, not applicable here yet)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = list(map(int, labels))
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        image_id = torch.tensor([index])
        target['image_id'] = image_id

        # apply transformers
        if self.transforms:
            transformed_image = self.transforms(image=image_resized,
                                                bboxes=target['boxes'],
                                                labels=labels)
            image_resized = transformed_image['image']
            target['boxes'] = torch.Tensor(transformed_image['bboxes'])

        return image_resized, target
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.images)

    @staticmethod
    def extract_bboxes_from_xml(bboxes_path: str,
                            class_label_name: str = 'class_no') -> Tuple[list, list]:
        """
        Extract bounding boxes and labels from an XML file.

        Args:
            bboxes_path (str):
                Path to the XML file containing bounding boxes.
                Default = None.
            class_label_name (str, optional):
                Name of the tag of the class label in the XML file.
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

    @staticmethod
    def resize_boxes(box: list,
                orig_height: int,
                orig_width: int,
                new_height: int,
                new_width: int) -> list:
        """
        Adjust the coordinates of a bounding box depending on the
        size adjustments of the underlying image.

        Args:
            box (list):
                List of coordinates [xmin, ymin, xmax, ymax].
            orig_height (int):
                Original height of the image.
            orig_width (int):
                Original width of the image.
            new_height (int):
                Target height after resizing.
            new_width (int):
                Target width after resizing.

        Returns:
            (list):
                Adjusted coordinates [xmin, ymin, xmax, ymax].
        """
        xmin = (box[0]/orig_width)*new_width
        ymin = (box[1]/orig_height)*new_height
        xmax = (box[2]/orig_width)*new_width
        ymax = (box[3]/orig_height)*new_height

        return [int(xmin), int(ymin), int(xmax), int(ymax)]
