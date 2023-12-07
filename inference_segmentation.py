"""Inference Script for the Segmentation Prediction setting.
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import os

from Segmentation.Custom3DSegmentationDataset import Custom3DSegmentationDataset
from Segmentation.utils import precision_recall_f1score
    

######## CONFIG ########
image_dir = './Data/SegmentationImages/'
run_name = 'prediction_run' # '20231122_2109_Pfronstetten_loc2_prediction'
model = 'prediction_model' # 'epoch_148_model.pth'
test_location = 'loc_name' # 'Pfronstetten_loc2'
image_format = 'png'
batch_size = 1
cmap = mcolors.ListedColormap(['#540c58', '#53c65a', '#ffa500', '#000000'])
######## CONFIG ########

print(f"""Start Inference {run_name} ✓""")

# make output paths
save_path = './Output/' + run_name + '/Inference/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"""Device: {device} ✓""")

# make dataset
test_dataset = Custom3DSegmentationDataset(
    data_dir=image_dir,
    data_list=[test_location],
    resize_to=(1024, 1024)
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1
)
if len(test_dataset) != 1:
    raise ValueError('Dataset Wrong')


# extract images and target mask
img_ts, mask_target = next(iter(test_loader))

# plot for visual sanity check
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img_ts[0, :, 0, :, :].permute(1, 2, 0))
ax[0].axis('off')
ax[1].imshow(img_ts[0, :, 1, :, :].permute(1, 2, 0))
ax[1].axis('off')
ax[2].imshow(mask_target.permute(1, 2, 0), cmap=cmap)
ax[2].axis('off')
plt.savefig((save_path + 'visual_sanity_check.png'), bbox_inches='tight', dpi=500)

# read in model
model_path = f'./Output/{run_name}/models/{model}'
model = torch.load(model_path, map_location=torch.device(device))
model.eval()
print(f"""Model loaded ✓""")


# make predictions
print(f"""Start Prediction ...""")
with torch.no_grad():
    images = img_ts.to(device)
    pred = model(images)
    class_predictions = torch.argmax(pred, dim=1)
print(f"""Prediction Done ✓""")


# plot predictions and target
fig, ax = plt.subplots(1, 2)
ax[0].imshow(class_predictions.numpy()[0], cmap=cmap)
ax[0].axis('off')
ax[1].imshow(mask_target.permute(1, 2, 0).numpy(), cmap=cmap)
ax[1].axis('off')
plt.savefig((save_path + 'pred_vs_true.png'), bbox_inches='tight', dpi=500)

# check for number of correct pixels
correct_pixels = (class_predictions == mask_target).sum()
pixel_acc = (correct_pixels / (1024*1024)).numpy()

# check for precision / recall / distribution
res_df = precision_recall_f1score(true_mask=mask_target.numpy(), pred_mask=class_predictions.numpy())
res_df = pd.concat([res_df, pd.Series(['pixel_accuracy', pixel_acc, 0, 0, 0])], axis=1)
res_df.to_csv(save_path + 'results.csv', index=False)

print(f"""Inference {run_name} Done ✓""")
