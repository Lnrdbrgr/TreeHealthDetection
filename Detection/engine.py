"""Script for model training.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import torch
import os

from models import create_FasterRCNN_model
from transformations import test_train_transforms, validation_transforms
from utils import create_dataloader, evaluate_loss, train_one_epoch

######## CONFIG ########
model = create_FasterRCNN_model(3)
learning_rate=0.0001
weight_decay=0.0005
num_epochs = 15
output_save_dir = 'Output'
run_name = str(datetime.now().strftime("%Y%m%d_%H%M"))
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

    # store training and validation loss
    training_loss.append(evaluate_loss(model, train_loader, device))
    validation_loss.append(evaluate_loss(model, validation_loader, device))

    # train the model
    train_one_epoch(model, train_loader, device, optimizer)

    # Response
    print(f"""Epoch: {epoch}
          Training Loss: {training_loss[-1]}
          Validation Loss: {validation_loss[-1]}""")

    if epoch == (num_epochs-1):
        save_direc = os.path.join(os.getcwd(), output_save_dir, run_name)
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)
        torch.save(model, os.path.join(save_direc, 'model.pth'))
        print(f"""Model Saved ✓""")
        loss_df = pd.DataFrame({'Epoch': range(len(training_loss)),
                                'TrainingLoss': training_loss,
                                'ValidationLoss': validation_loss})
        loss_df.to_csv(os.path.join(save_direc, 'loss_df.csv'))
        print(f"""Loss Saved ✓""")
