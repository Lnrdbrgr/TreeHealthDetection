"""
This module provides functions for image visualization.
"""

import copy
import cv2
from typing import Tuple
from xml.etree import ElementTree as et
import matplotlib.pyplot as plt


def show_image(image_path: str,
               bboxes_path: str = None,
               class_label_name: str = 'name',
               resize_to: Tuple[int, int] = None,
               transformations: any = None,
               show_before_after: bool = False,
               env_google_colab: bool = False) -> None:
    """
    Display an image with optional bounding boxes and transformations.

    Args:
        image_path (str):
            Path to the image file.
        bboxes_path (str, optional):
            Path to the XML file containing bounding boxes.
            Default = None.
        class_label_name (str, optional):
            Name of the class label in the XML file.
            Default = 'name'.
        resize_to (Tuple[int, int], optional):
            Target size for resizing the image.
            Default = None.
        transformations (any, optional):
            Image transformation function.
            Defaults to None.
        show_before_after (bool, optional):
            Whether to display a before/after comparison.
            Default = False.
        env_google_colab (bool, optional):
            Flag indicating if the code is running in Google Colab environment.
            Default = False.

    Returns:
        None
    """
    # check if environment is Google Colab as opencv behaves different
    if env_google_colab:
        from google.colab.patches import cv2_imshow
    # get image
    image = cv2.imread(image_path)
    # get bounding boxes if applicable
    if bboxes_path:
        # get boxes
        boxes, labels = extract_bboxes_from_xml(
            bboxes_path=bboxes_path,
            class_label_name=class_label_name
        )
        # resize the bounding boxes according to desired height/width
        if resize_to:
            new_boxes = []
            for box in boxes:
                new_box = resize_boxes(box,
                                       orig_height=image.shape[0],
                                       orig_width=image.shape[1],
                                       new_height=resize_to[0],
                                       new_width=resize_to[1])
                new_boxes.append(new_box)
            boxes = new_boxes
    # resize
    if resize_to:
        image = cv2.resize(image, (resize_to[0], resize_to[1]))
    # show if before/after comparison
    if show_before_after:
        show_image_before = copy.deepcopy(image)
        if bboxes_path:
            for box in boxes:
                cv2.rectangle(show_image_before,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color=(0, 0, 255), thickness=2)
        if env_google_colab:
            cv2_imshow(show_image_before)
        else:
            plt.figure()
            plt.imshow(show_image_before)
            plt.show()
            # Some Trouble with openCV and imshow
            #cv2.imshow('Image', show_image_before)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    # transform if applicable
    if transformations:
        image = image/255.0
        image = transformations(image=image,
                                bboxes=boxes,
                                labels=labels)
        # commented out with transition from imshow to plt
        #image['image'] = image['image']*255.0
        show_image = image['image']
        boxes = image['bboxes']
    else:
        show_image = image
    # show image
    if bboxes_path:
        for box in boxes:
            cv2.rectangle(show_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color=(0, 0, 255), thickness=2)
    if env_google_colab:
        cv2_imshow(show_image)
    else:
        plt.figure()
        plt.imshow(show_image)
        plt.show()
        # Some Trouble with openCV and imshow
        #cv2.imshow('Image', show_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


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
        resize_to (tuple, optional):
            Target size for resizing the image.
            Default = None.

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
        list: Adjusted coordinates [xmin, ymin, xmax, ymax].
    """
    xmin = (box[0]/orig_width)*new_width
    ymin = (box[1]/orig_height)*new_height
    xmax = (box[2]/orig_width)*new_width
    ymax = (box[3]/orig_height)*new_height

    return [int(xmin), int(ymin), int(xmax), int(ymax)]
