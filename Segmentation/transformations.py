"""Module to store transform variations.
"""

import albumentations as A

# transformations for serious runs, needs to be ReplayCompose for 3D inputs
train_transforms = A.ReplayCompose([
    A.RandomRotate90(p=0.75),
    A.VerticalFlip(p=0.75),
    A.HorizontalFlip(p=0.75)
])