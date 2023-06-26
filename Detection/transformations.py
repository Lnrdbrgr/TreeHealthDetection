"""Module to store transform variations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# transformations for test runs
test_train_transforms = A.Compose([
                  A.VerticalFlip(p=0.5),
                  A.HorizontalFlip(p=0.5),
                  ToTensorV2(p=1.0),
              ], bbox_params={
                  'format': 'pascal_voc',
                  'label_fields': ['labels']
              })

# transformations for the validation set
validation_transforms = A.Compose([
                  ToTensorV2(p=1.0),
              ], bbox_params={
                  'format': 'pascal_voc',
                  'label_fields': ['labels']
              })