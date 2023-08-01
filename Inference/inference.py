"""This module provides functions to evaluate the performance of a
trained object detection model on a test dataset. The evaluation
includes calculating the Mean Average Precision (MAP) overall and per
class. Additionally, the class distribution of the test set is stored
in a CSV file for further analysis.
"""

from collections import Counter
from itertools import chain
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from CustomDataset import CustomDataset
from transformations import validation_transforms
from utils import collate_fn, append_dicts, transform_dict

######## CONFIG ########
image_dir = '../Data/ProcessedImages'
xml_dir = '../Data/ProcessedImages'
run_name = '20230728_0858_hbgloc1fasterrcnnresnet'
model = 'epoch_140_model.pth'
test_pattern = 'Hachenburg_loc1'
label_mapping_dict = {'_background_': 0, 'healthy': 1, 'infested': 2, 'dead': 3}
image_format = 'png'
batch_size = 1
######## CONFIG ########

# invert mapping dict for convenient naming later on
inv_mapping_dict = {v: k for k, v in label_mapping_dict.items()}

# get all test images
images = [image for image in os.listdir(image_dir) \
                if image.startswith(test_pattern) and \
                image.endswith(image_format)]

# create Dataset and DataLoader
test_dataset = CustomDataset(image_dir=image_dir,
                             image_list=images,
                             label_mapping_dict=label_mapping_dict,
                             xml_dir=xml_dir,
                             image_format=image_format,
                             height=512,
                             width=512,
                             transforms=validation_transforms)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)
print(f"""DataLoader initialized with {len(test_dataset)} images in test set ✓""")


# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"""Device: {device} ✓""")

# read in model
model_path = f'Output/{run_name}/Models/{model}'
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
print(f"""Model loaded ✓""")

# initialize loop objects
predictions = []
targets = []

# make predictions
print(f"""Make Predictions:""")
for images, ground_truth in tqdm(test_loader):
    images = [image.to(device) for image in images]
    with torch.no_grad():
        help_predictions = model(images)
    # append for final object
    predictions.extend(help_predictions) 
    targets.extend(list(ground_truth))
print(f"""Predictions on Test Set done ✓""")

# check accuracy: overall
MAPOverall = MeanAveragePrecision()
MAPOverall.update(preds=predictions, target=targets)
map_overall = MAPOverall.compute()
print(f"""MAP Overall done ✓""")

# check accuracy: per class
# initialize loop objects
classes = [value for key, value in label_mapping_dict.items() if key != '_background_']
map_per_class = {}

for filter_class in classes:
    # initialize loop objects
    class_predictions = []
    class_targets = []
    # filter on classes and extract predictions of interest
    for prediction in predictions:
        selected_boxes = [box.tolist() for box, label in zip(prediction['boxes'], prediction['labels']) if label == filter_class]
        selected_labels = [label.item() for label in prediction['labels'] if label == filter_class]
        selected_scores = [pred_score.item() for pred_score, label in zip(prediction['scores'], prediction['labels']) if label == filter_class]
        help_dict = {'boxes': torch.tensor(selected_boxes),
                    'labels': torch.tensor(selected_labels),
                    'scores': torch.tensor(selected_scores)}        
        class_predictions.append(help_dict)
    # extract targets of interest
    for target in targets:
        selected_boxes = [box.tolist() for box, label in zip(target['boxes'], target['labels']) if label == filter_class]
        selected_labels = [label.item() for label in target['labels'] if label == filter_class]
        help_dict_targets = {'boxes': torch.tensor(selected_boxes),
                    'labels': torch.tensor(selected_labels)}
        class_targets.append(help_dict_targets)
    # calculate MAP
    MAPClass = MeanAveragePrecision()
    MAPClass.update(preds=class_predictions, target=class_targets)
    map_class = MAPClass.compute()
    append_dicts(map_overall, map_class)
print(f"""MAP per Class done ✓""")

# make dataframe
map_overall = transform_dict(map_overall)
map_overall_pd = pd.DataFrame.from_dict(map_overall, orient='index') \
    .reset_index()
columns = ['Metric', 'MAP_Overall']
for i in classes:
    columns.append(
        f'MAP_{inv_mapping_dict[i]}'
    )    
map_overall_pd.columns = columns
map_overall_pd.to_csv(f'Output/{run_name}/test_MAP.csv', index=False)
print(f"""MAP Data saved ✓""")

# Store class distribution in test set
labels = list(chain.from_iterable([target['labels'].tolist() for target in targets]))
test_set_class_distribution = pd.DataFrame([Counter(labels)]) \
    .rename(columns=inv_mapping_dict)
test_set_class_distribution.to_csv(f'Output/{run_name}/test_set_class_distribution.csv',
                                   index=False)
print(f"""Training Set class distribution saved ✓""")
