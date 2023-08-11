"""Script for model training.
"""

import copy
from datetime import datetime
from pytz import timezone
import torch

from Detection.models import create_FasterRCNN_resnet50_model, \
    create_Retinanet_resnet50_v2_model, create_FasterRCNN_mobilenet_v3_model, \
    create_ssd300_vgg16_model
from Detection.transformations import train_transforms,  test_train_transforms, \
    validation_transforms
from Detection.utils import create_dataloader, evaluate_loss, train_one_epoch, \
    write_out_results, write_out_model, append_dicts, visualize_training_output
from Detection.evaluation_utils import evaluate_MAP

######## CONFIG ########
model = create_FasterRCNN_resnet50_model(3)
learning_rate=0.0001
weight_decay=0.0005
num_epochs = int(input('Number of Epochs: '))
test_pattern = 'Haiterbach_loc1'
output_save_dir = 'Output'
run_name = str(datetime.now(timezone('Europe/Berlin')).strftime("%Y%m%d_%H%M"))
train_transformations = train_transforms
label_mapping_dict={'_background_': 0, 'healthy': 1, 'infested': 2, 'dead': 3}
######## CONFIG ########

# create dataloader
train_loader, validation_loader, train_val_images_dict = create_dataloader(
    train_img_directory='./Data/ProcessedImages',
    train_xml_directory='./Data/ProcessedImages',
    label_mapping_dict=label_mapping_dict,
    train_dir_is_valid_dir=True,
    test_pattern=test_pattern,
    train_transforms=train_transformations,
    validation_transforms=validation_transforms,
    train_batch_size=8,
    train_split=0.8
)
print(f"""Training and Validation Data Loader initialized ✓""")

# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"""Device: {device} ✓""")

# initialize model
model = model.to(device)

# initialize optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=learning_rate,
                              weight_decay=weight_decay)

# initialize learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=25,
                                               gamma=0.9)

# initialize loop objects
training_loss = []
validation_loss = []
training_MAP_dict = {}
validation_MAP_dict = {}

print(f"""Length Training Set: {len(train_loader.dataset)}""")
print(f"""Length Validation Set: {len(validation_loader.dataset)}""")
print(f"""Start Training with {num_epochs} epochs ✓""")

# train the model
for epoch in range(num_epochs):

    # train the model
    train_one_epoch(model, train_loader, device, optimizer)

    # update the learning rate
    lr_scheduler.step()

    # store evaluation metrics
    eval_model = copy.deepcopy(model)
    training_loss.append(evaluate_loss(eval_model, train_loader, device))
    validation_loss.append(evaluate_loss(eval_model, validation_loader, device))    
    train_map = evaluate_MAP(model=eval_model, dataloader=train_loader, device=device)
    append_dicts(training_MAP_dict, train_map)
    validation_map = evaluate_MAP(model=eval_model, dataloader=validation_loader, device=device)
    append_dicts(validation_MAP_dict, validation_map)
        
    # write out the model if better than previous model
    if (epoch > 5) and \
       ((validation_MAP_dict['map'][-1] >= max(validation_MAP_dict['map'])) \
        or (validation_MAP_dict['map_50'][-1] >= max(validation_MAP_dict['map_50']))):
        write_out_model(
            model=model,
            output_directory=output_save_dir,
            run_name=run_name,
            epoch=epoch
        )

    # write out or update results
    if (epoch == 1):
        write_out_results(
            output_directory=output_save_dir,
            run_name=run_name,
            model=model,
            train_transformations=train_transformations,
            optimizer=optimizer,
            learning_rate_scheduler=lr_scheduler,
            write_out_dicts={'TrainValSplit': train_val_images_dict}
        )
    elif (epoch > 5):
        write_out_results(
            output_directory=output_save_dir,
            run_name=run_name,
            training_loss=training_loss,
            validation_loss=validation_loss,
            training_MAP=training_MAP_dict,
            validation_MAP=validation_MAP_dict
        )
        visualize_training_output(output_folder=run_name)

    if epoch > 1:
        # Response
        print(f"""Epoch: {epoch}
              Training Loss: {training_loss[-5:]}
              Validation Loss: {validation_loss[-5:]}
              Training MAP50:95: {training_MAP_dict['map'][-5:]}
              Validation MAP50:95: {validation_MAP_dict['map'][-5:]}
              Training MAP50: {training_MAP_dict['map_50'][-5:]}
              Validation MAP50: {validation_MAP_dict['map_50'][-5:]}""")
    else:
        print(f"""Epoch: {epoch} ✓""")
