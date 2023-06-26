"""Module for utility and helper functions during the object detection
and model training phase.
"""

from xml.etree import ElementTree as et


def extract_bboxes_from_xml(bboxes_path: str,
                            class_label_name: str = 'name') -> tuple[list, list]:
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
        tuple[list, list]: Bounding boxes and labels as lists.
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
        (list):
            Adjusted coordinates [xmin, ymin, xmax, ymax].
    """
    xmin = (box[0]/orig_width)*new_width
    ymin = (box[1]/orig_height)*new_height
    xmax = (box[2]/orig_width)*new_width
    ymax = (box[3]/orig_height)*new_height

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


