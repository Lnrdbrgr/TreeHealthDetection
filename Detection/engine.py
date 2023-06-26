"""
"""


from transformations import test_train_transforms, validation_transforms
from utils import create_dataloader

# create dataloader
train_loader, validation_loader = create_dataloader(
    train_img_directory='../Data/ProcessedImages',
    train_xml_directory='../Data/ProcessedImagesXMLs',
    train_dir_is_valid_dir=True,
    train_transforms=test_train_transforms,
    validation_transforms=validation_transforms
)

images, targets = next(iter(train_loader))
print(images[0])


