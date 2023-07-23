"""Module to examplarly plot an image with bounding boxes.
"""

import albumentations as A

from visualization import show_image


# specify some transformations if desired
transforms = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    #A.RandomSizedCrop((220, 350), 512, 512, p=0.2),
], bbox_params={
    'format': 'pascal_voc',
    'label_fields': ['labels']
})


# show image
show_image(image_path='../Data/ProcessedImages/example_image_01.png',
           bboxes_path='../Data/ProcessedImagesXMLs/example_image_01.xml',
           #transformations=transforms,
           #show_before_after=True,
           class_label_name='class_label')
