"""
"""

import numpy as np
import torch

from models import create_FasterRCNN_model
from transformations import test_train_transforms, validation_transforms
from utils import create_dataloader

######## CONFIG ########
model = create_FasterRCNN_model(3)
learning_rate=0.0001
weight_decay=0.0005
num_epochs = 15
######## CONFIG ########

# create dataloader
train_loader, validation_loader = create_dataloader(
    train_img_directory='../Data/ProcessedImages',
    train_xml_directory='../Data/ProcessedImagesXMLs',
    train_dir_is_valid_dir=True,
    train_transforms=test_train_transforms,
    validation_transforms=validation_transforms,
    train_batch_size=8
)
print(f"""Training and Validation Data Loader initialized ✓""")

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"""Device: {device}""")

# initiate model
model = model.to(device)

# initiate optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

# initiate loop objects
training_loss = []
validation_loss = []
train_MAP = []
validation_MAP = []


# train the model
for epoch in range(num_epochs):

    val_loss = []
    train_loss = []
    # check model performance on validation set
    for data in validation_loader:
        # set the model to eval modus
        model.eval()
        # extract the data
        images, target = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]
        # extract the losses
        with torch.no_grad():
            val_loss_dict = model(images, targets)
            val_loss.append(sum(loss for loss in val_loss_dict.values()))
    validation_loss.append(np.mean(val_loss))
        

    for data in train_loader:            
        # set the model to training modus
        model.train()
        # set the optimizer
        optimizer.zero_grad()
        # extract the data
        images, target = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]

        # extract the losses
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        with torch.no_grad():
            train_loss.append(loss)

        # do the magic
        loss.backward()
        optimizer.step()
    train_loss.append(np.mean(train_loss))

    # Response
    print(f'Epoch: {epoch}')
    print(train_loss)

    if epoch == (num_epochs-1):
        torch.save(model, './SavedModels/model.pth')
        print(f"""Model Saved ✓""")

        # Bug somewhere, doesnt print in Colab
        print(validation_loss)
        print(training_loss)
