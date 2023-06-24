"""Module for utility and helper functions during the pre-processing
of the image data.
"""

import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree


def calculate_area(x_min: int, x_max: int, y_min: int, y_max: int) -> int:
    """
    Calculate the area in pixels given the coordinates of a bounding box.

    Args:
        x_min (int):
            Minimum x-coordinate of the bounding box.
        x_max (int):
            Maximum x-coordinate of the bounding box.
        y_min (int):
            Minimum y-coordinate of the bounding box.
        y_max (int):
            Maximum y-coordinate of the bounding box.

    Returns:
        int: The calculated area in pixels.
    """
    area = (x_max - x_min) * (y_max - y_min)
    return area


def write_PascalVOC_xml(filename: str,
                        folder: str,
                        path: str,
                        data_source: str,
                        img_height: int,
                        img_width: int,
                        bbox_df: pd.DataFrame,
                        XML_folder: str,
                        img_depth: int = 3,
                        segmented: int = 0) -> None:
    """
    Write Pascal VOC XML file for an image with bounding boxes.

    Reference:
    bit.ly/46laMHl, bit.ly/3PtUo1x

    Args:
        filename (str):
            Name of the image file.
        folder (str):
            Name of the folder containing the image file.
        path (str):
            Path to the image file.
        data_source (str):
            Source of the data.
        img_height (int):
            Height of the image.
        img_width (int):
            Width of the image.
        bbox_df (pd.DataFrame):
            DataFrame containing bounding box information.
        XML_folder (str):
            Folder to save the XML file.
        img_depth (int, optional):
            Depth of the image, 3 channels (RGB) correspond to a depth
            of 3.
            Default = 3
        segmented (int, optional):
            Indicates if the image is segmented.
            Default = 3

    Returns:
        None:
            This function does not return a value but writes out the
            desired XML in the specified folder.
    """

    # create xml root element
    root = Element('annotation')
    # add specifications
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = path
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = data_source
    size = SubElement(root, 'size')
    SubElement(size, 'height').text = str(img_height)
    SubElement(size, 'width').text = str(img_width)
    SubElement(size, 'depth').text = str(img_depth)
    SubElement(root, 'segmented').text = str(segmented)

    # add bounding boxes
    for _, row in bbox_df.iterrows():
        # make a bounding box object
        obj = SubElement(root, 'object')
        # assign class label
        SubElement(obj, 'name').text = row['health_status']
        # assign optional Pascal VOC specifications
        if 'pose' in bbox_df.columns:
            SubElement(obj, 'pose').text = row['pose']
        else:
            SubElement(obj, 'pose').text = 'Unspecified'
        if 'truncated' in bbox_df.columns:
            SubElement(obj, 'truncated').text = row['truncated']
        else:
            SubElement(obj, 'truncated').text = '0'
        if 'difficult' in bbox_df.columns:
            SubElement(obj, 'difficult').text = row['difficult']
        else:
            SubElement(obj, 'difficult').text = '0'
        # assign bounding box
        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(row['x_min'])
        SubElement(bbox, 'ymin').text = str(row['y_min'])
        SubElement(bbox, 'xmax').text = str(row['x_max'])
        SubElement(bbox, 'ymax').text = str(row['y_max'])

    # create the tree
    tree = ElementTree(root)
    # write the xml file
    xml_filename = XML_folder + filename.split('.')[0] + '.xml'
    tree.write(xml_filename, encoding="utf-8")
