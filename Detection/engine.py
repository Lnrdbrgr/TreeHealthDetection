"""Script for model training.
"""

import copy
from datetime import datetime
import torch

from models import create_FasterRCNN_model
from transformations import train_transforms,  test_train_transforms, \
    validation_transforms
from utils import create_dataloader, evaluate_loss, train_one_epoch, \
    write_out_results, append_dicts
from evaluation_utils import evaluate_MAP

######## CONFIG ########
model = create_FasterRCNN_model(3)
learning_rate=0.0001
weight_decay=0.0005
num_epochs = int(input('Number of Epochs: '))
output_save_dir = 'Output'
run_name = str(datetime.now().strftime("%Y%m%d_%H%M"))
train_transformations = train_transforms
######## CONFIG ########

# create dataloader
train_loader, validation_loader = create_dataloader(
    train_img_directory='../Data/ProcessedImages',
    train_xml_directory='../Data/ProcessedImagesXMLs',
    train_dir_is_valid_dir=True,
    train_transforms=train_transformations,
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
optimizer = torch.optim.AdamW(params, lr=learning_rate,
                              weight_decay=weight_decay)

# initiate loop objects
training_loss = []
validation_loss = []
training_MAP_dict = {}
validation_MAP_dict = {}

print(f"""Length Training Set: {len(train_loader.dataset)}""")
print(f"""Length Validation Set: {len(validation_loader.dataset)}""")
print(f"""Start Training ✓""")

# train the model
for epoch in range(num_epochs):

    # train the model
    train_one_epoch(model, train_loader, device, optimizer)

    # store evaluation metrics
    eval_model = copy.deepcopy(model)
    training_loss.append(evaluate_loss(eval_model, train_loader, device))
    validation_loss.append(evaluate_loss(eval_model, validation_loader, device))    
    train_map = evaluate_MAP(model=eval_model, dataloader=train_loader, device=device)
    append_dicts(training_MAP_dict, train_map)
    validation_map = evaluate_MAP(model=eval_model, dataloader=validation_loader, device=device)
    append_dicts(validation_MAP_dict, validation_map)

    # Response
    print(f"""Epoch: {epoch}
          Training Loss: {training_loss}
          Validation Loss: {validation_loss}
          Training MAP50:95: {training_MAP_dict['map'][-5:]}
          Validation MAP50:95: {validation_MAP_dict['map'][-5:]}
          Training MAP50: {training_MAP_dict['map_50'][-5:]}
          Validation MAP50: {validation_MAP_dict['map_50'][-5:]}""")

    # write out model after every second epoch
    if (epoch%2 == 0):
        write_out_results(
                model=model,
                output_directory=output_save_dir,
                run_name=run_name,
                epoch=epoch
        )
    # save results on last epoch
    if epoch == (num_epochs-1):
        write_out_results(
            model=model,
            output_directory=output_save_dir,
            run_name=run_name,
            epoch=epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
            training_MAP=training_MAP_dict,
            validation_MAP=validation_MAP_dict,
            optimizer=optimizer,
            train_transformations=train_transformations
        )
