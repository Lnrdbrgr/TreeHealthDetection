"""Script for model training.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import torch
import os

from models import create_FasterRCNN_model
from transformations import train_transforms,  test_train_transforms, \
    validation_transforms
from utils import create_dataloader, evaluate_loss, train_one_epoch, \
    write_out_results
from evaluation_utils import evaluate_coco_MAP

######## CONFIG ########
model = create_FasterRCNN_model(3)
learning_rate=0.0001
weight_decay=0.0005
num_epochs = int(input('Number of Epochs: '))
output_save_dir = 'Output'
run_name = str(datetime.now().strftime("%Y%m%d_%H%M"))
######## CONFIG ########

# create dataloader
train_loader, validation_loader = create_dataloader(
    train_img_directory='../Data/ProcessedImages',
    train_xml_directory='../Data/ProcessedImagesXMLs',
    train_dir_is_valid_dir=True,
    train_transforms=train_transforms,
    validation_transforms=validation_transforms,
    train_batch_size=8
)
print(f"""Training and Validation Data Loader initialized ✓""")

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"""Device: {device} ✓""")

# initiate model
model = model.to(device)

# initiate optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

# initiate loop objects
training_loss = []
validation_loss = []
training_MAP = []
validation_MAP = []

# train the model
for epoch in range(num_epochs):

    # store training and validation loss
    training_loss.append(evaluate_loss(model, train_loader, device))
    validation_loss.append(evaluate_loss(model, validation_loader, device))

    # store training and validation MAP
    print('MAP validation started - training')
    training_MAP.append(evaluate_coco_MAP(model, train_loader, device))
    print('MAP validation started - validation')
    validation_MAP.append(evaluate_coco_MAP(model, validation_loader, device))
    print('Done')

    # train the model
    train_one_epoch(model, train_loader, device, optimizer)

    # Response
    print(f"""Epoch: {epoch}
          Training Loss: {training_loss[-1]}
          Validation Loss: {validation_loss[-1]}""")

    # save results on last epoch
    if epoch == (num_epochs-1):
        write_out_results(
            model=model,
            output_directory=output_save_dir,
            run_name=run_name,
            training_loss=training_loss,
            validation_loss=validation_loss,
            optimizer=optimizer
        )
