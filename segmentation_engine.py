"""
"""

import torch
import cv2
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from pytz import timezone
import copy

from Segmentation.Models.UNet3DPreTrained import UNet3DPreTrained
from Segmentation.Custom3DSegmentationDataset import Custom3DSegmentationDataset
from Detection.utils import visualize_training_output
from Segmentation.utils import create_dataloader, pixel_accuracy, jaccard_accuracy, write_out_results, evaluate_segmentation_accuracy
from Segmentation.transformations import train_transforms

######## CONFIG ########
learning_rate=0.01
weight_decay=0.0005
num_epochs = int(input('Number of Epochs: '))
run_name = str(datetime.now(timezone('Europe/Berlin')).strftime("%Y%m%d_%H%M"))
model = UNet3DPreTrained(in_channels=3, out_channels=4,
                         input_size=512, output_size=512, t=2)
output_save_dir = './Output/'
data_directory ='./Data/SegmentationImages/'
train_transforms = train_transforms
######## CONFIG ########


# make dataset
train_loader, validation_loader = create_dataloader(
    data_directory=data_directory,
    random_train_val=True,
    resize_to=(512, 512),
    train_transforms=train_transforms,
    test_images=None
)
print(f"""Training and Validation Data Loader initialized ✓""")

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"""Device: {device} ✓""")

# model
model = model.to(device)

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
loss_fn = torch.nn.CrossEntropyLoss()

# initialize loop objects
training_loss = []
validation_loss = []
training_pixel_acc = []
validation_pixel_acc = []
training_jaccard_acc = []
validation_jaccard_acc = []

print(f"""Length Training Set: {len(train_loader.dataset)}""")
print(f"""Length Validation Set: {len(validation_loader.dataset)}""")
print(f"""Start Training with {num_epochs} epochs ✓""")

# train model
for epoch in range(num_epochs):
    model.train()
    for images, target in train_loader:
        # set the optimizer
        optimizer.zero_grad()
        # move tho respective device cpu/gpu
        images = images.to(device)
        target = target.to(device)
        # make predictions
        preds = model(images)
        loss = loss_fn(preds, target)
        # do the magic
        loss.backward()
        optimizer.step()

    # update the learning rate
    lr_scheduler.step()

    # store metrics
    eval_model = copy.deepcopy(model)
    train_loss, train_pixel_acc, train_jaccard_acc = evaluate_segmentation_accuracy(
        model=eval_model,
        data_loader=train_loader,
        device=device,
        loss_fn=loss_fn
    )
    val_loss, val_pixel_acc, val_jaccard_acc = evaluate_segmentation_accuracy(
        model=eval_model,
        data_loader=validation_loader,
        device=device,
        loss_fn=loss_fn
    )
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
    training_pixel_acc.append(train_pixel_acc)
    validation_pixel_acc.append(val_pixel_acc)
    training_jaccard_acc.append(train_jaccard_acc)
    validation_jaccard_acc.append(val_jaccard_acc)

    # write out results
    if epoch == 5:
        write_out_results(output_directory=output_save_dir,
                          run_name=run_name,
                          training_loss=training_loss,
                          validation_loss=validation_loss,
                          training_pixel_acc=training_pixel_acc,
                          validation_pixel_acc=validation_pixel_acc,
                          training_jaccard=training_jaccard_acc,
                          validation_jaccard=validation_jaccard_acc,
                          optimizer=optimizer,
                          learning_rate_scheduler=lr_scheduler,
                          model=model)
    else:
        write_out_results(output_directory=output_save_dir,
                          run_name=run_name,
                          training_loss=training_loss,
                          validation_loss=validation_loss,
                          training_pixel_acc=training_pixel_acc,
                          validation_pixel_acc=validation_pixel_acc,
                          training_jaccard=training_jaccard_acc,
                          validation_jaccard=validation_jaccard_acc)

    # verbose
    print(f"""Epoch: {epoch} ✓
          Training Loss: {training_loss[-5:]}
          Validation Loss: {validation_loss[-5:]}
          Training Pixel Accuracy: {training_pixel_acc[-5:]}
          Validation Pixel Accuracy: {validation_pixel_acc[-5:]}
          Training Jaccard: {np.mean(training_jaccard_acc[-5:], axis=1)}
          Validation Jaccard: {np.mean(validation_jaccard_acc[-5:], axis=1)}""")
    