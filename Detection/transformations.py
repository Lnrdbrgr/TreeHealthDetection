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

# transformations for serious runs
train_transforms = A.Compose([
    A.RandomRotate90(p=0.75),
    A.VerticalFlip(p=0.75),
    A.HorizontalFlip(p=0.75),
    #A.RandomSizedCrop((450, 450), 512, 512, p=0.1),
    A.Sharpen(p=0.1, lightness=(0.2, 0.8)),
    # the following transformations really distort the image so dont
    # use them that often
    A.OneOf([
        A.MotionBlur(p=1),
        A.Blur(blur_limit=3, p=1),
        A.RandomSnow(p=1, brightness_coeff=1.8, snow_point_upper=0.2),
        A.RandomRain(p=1, slant_lower=-4, slant_upper=4, drop_length=20,
                     blur_value=3, brightness_coefficient=0.8),
        A.RandomShadow(p=1)
    ], p=0.1),
    ToTensorV2(p=1.0)
], bbox_params={
    'format': 'pascal_voc',
    'label_fields': ['labels']
})