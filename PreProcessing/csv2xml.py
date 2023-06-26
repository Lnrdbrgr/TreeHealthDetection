"""Module to translate a csv to xml files.

This module contains the functionality to translate a csv file with
image names and bounding boxes to xml files in Pascal Visual Object
Classes (VOC) standard.
"""

import pandas as pd

from utils import write_PascalVOC_xml

######## CONFIG ########
annotation_csv = '../Data/ProcessedImagesCSVs/file.csv'
image_folder = 'ProcessedImagesXMLs/'
image_path = '../Data/ProcessedImagesXMLs/'
data_source = 'source'
XML_output_folder='../Data/ProcessedImagesXMLs/'
img_size = 512
######## CONFIG ########

if __name__ == '__main__':

    # read in annotation csv
    annotation_csv = pd.read_csv(annotation_csv)

    # extract images
    images = annotation_csv['image'].unique()

    # write out the XML for each image
    for image in images:
        write_PascalVOC_xml(filename=str(image),
                            folder=image_folder,
                            path=image_path,
                            data_source=data_source,
                            img_height=img_size,
                            img_width=img_size,
                            bbox_df=annotation_csv.query('image == @image'),
                            XML_folder=XML_output_folder)
        
    print('Done ✓')
