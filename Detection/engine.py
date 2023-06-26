"""
"""

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


# train the model
for epoch in range(num_epochs):
    # set the model to training modus
    model.train()
    for data in train_loader:
        with torch.no_grad():
            pass
            # ToDo: evaluation with validation dataset
            # go through everything in validation loader, store and make average
            
        # set the optimizer
        optimizer.zero_grad()
        # extract the data
        images, target = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]

        # extract the losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # do the magic
        losses.backward()
        optimizer.step()

    # Response
    print(f'Epoch: {epoch}')
    print(losses)

    if epoch == (num_epochs-1):
        torch.save(model, './SavedModels/model.pth')
        print(f"""Model Saved ✓""")
