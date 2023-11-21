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
import pandas as pd

from Segmentation.Models.UNet3DPreTrained import UNet3DPreTrained
from Segmentation.Custom3DSegmentationDataset import Custom3DSegmentationDataset
from Detection.utils import visualize_training_output
from Segmentation.utils import create_dataloader, write_out_results, evaluate_segmentation_accuracy, write_out_model
from Segmentation.transformations import train_transforms

######## CONFIG ########
test_images = 'Pfronstetten_loc1'
learning_rate=0.001
weight_decay=0.0005
num_epochs = int(input('Number of Epochs: '))
run_name = str(datetime.now(timezone('Europe/Berlin')).strftime("%Y%m%d_%H%M")) + \
    '_' + test_images
resize_to = 64 #1024
model = UNet3DPreTrained(in_channels=3, out_channels=4,
                         input_size=resize_to, output_size=resize_to, t=2)
output_save_dir = './Output/'
data_directory ='./Data/SegmentationImages/'
train_transforms = None
######## CONFIG ########


# make dataset
train_loader, validation_loader = create_dataloader(
    data_directory=data_directory,
    random_train_val=False,
    validation_images='vert+hor_flip',
    resize_to=(resize_to, resize_to),
    train_transforms=train_transforms,
    test_images=test_images
)
print(f"""Training and Validation Data Loader initialized ({resize_to}) ✓""")

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
training_metrics = pd.DataFrame()
validation_metrics = pd.DataFrame()

print(f"""Length Training Set: {len(train_loader.dataset)}""")
print(f"""Length Validation Set: {len(validation_loader.dataset)}""")
print(f"""Test Images: {test_images}""")
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
    train_loss, train_pixel_acc, train_metrics_df = evaluate_segmentation_accuracy(
        model=eval_model,
        data_loader=train_loader,
        device=device,
        loss_fn=loss_fn
    )
    val_loss, val_pixel_acc, val_metrics_df = evaluate_segmentation_accuracy(
        model=eval_model,
        data_loader=validation_loader,
        device=device,
        loss_fn=loss_fn
    )
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
    training_pixel_acc.append(train_pixel_acc)
    validation_pixel_acc.append(val_pixel_acc)
    train_metrics_df['epoch'] = epoch
    val_metrics_df['epoch'] = epoch
    training_metrics = pd.concat([training_metrics, train_metrics_df])
    validation_metrics = pd.concat([validation_metrics, val_metrics_df])


    # write out results
    if epoch == 5:
        write_out_results(output_directory=output_save_dir,
                          run_name=run_name,
                          training_loss=training_loss,
                          validation_loss=validation_loss,
                          training_pixel_acc=training_pixel_acc,
                          validation_pixel_acc=validation_pixel_acc,
                          training_metrics_df=training_metrics,
                          validation_metrics_df=validation_metrics,
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
                          training_metrics_df=training_metrics,
                          validation_metrics_df=validation_metrics,)
    # write out model if better than previous model
    if (epoch > 5) & (validation_pixel_acc[-1] >= max(validation_pixel_acc)):
        write_out_model(model=model,
                        output_directory=output_save_dir,
                        run_name=run_name,
                        epoch=epoch)

    # verbose
    print(f"""Epoch: {epoch} ✓
          Training Loss: {training_loss[-5:]}
          Validation Loss: {validation_loss[-5:]}
          Training Pixel Accuracy: {training_pixel_acc[-5:]}
          Validation Pixel Accuracy: {validation_pixel_acc[-5:]}
          Training Background F1-Score: {list(training_metrics.query('metric == "f1_score"').sort_values('epoch')['background'])[-5:]}
          Training Healthy F1-Score: {list(training_metrics.query('metric == "f1_score"').sort_values('epoch')['healthy'])[-5:]}
          Training Infested F1-Score: {list(training_metrics.query('metric == "f1_score"').sort_values('epoch')['infested'])[-5:]}
          Training Dead F1-Score: {list(training_metrics.query('metric == "f1_score"').sort_values('epoch')['dead'])[-5:]}
          Validation Background F1-Score: {list(validation_metrics.query('metric == "f1_score"').sort_values('epoch')['background'])[-5:]}
          Validation Healthy F1-Score: {list(validation_metrics.query('metric == "f1_score"').sort_values('epoch')['healthy'])[-5:]}
          Validation Infested F1-Score: {list(validation_metrics.query('metric == "f1_score"').sort_values('epoch')['infested'])[-5:]}
          Validation Dead F1-Score: {list(validation_metrics.query('metric == "f1_score"').sort_values('epoch')['dead'])[-5:]}""")
    