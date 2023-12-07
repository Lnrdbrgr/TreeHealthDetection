"""Script to visualize the metrics during training
"""

import pandas as pd
import matplotlib.pyplot as plt


######## CONFIG ########
path = './Output/' + 'run_CHANGE_HERE' + '/'

background_color: str = '#eff1f3'
train_color: str = '#222843'
validation_color: str = '#dda15e'
######## CONFIG ########

# read in data
validation_metrics = pd.read_csv(path + 'validation_metrics.csv')
train_metrics = pd.read_csv(path + 'training_metrics.csv')
loss_df = pd.read_csv(path + 'loss_df.csv')
pixel_acc = pd.read_csv(path + 'pixel_accuracy.csv')

# extract values of interest
metrics_dict = {}

for data, name in list(zip([train_metrics, validation_metrics], ['train', 'val'])):
    for status in ['healthy', 'infested', 'dead']:
        x = data.query('metric == "f1_score"').sort_values('epoch')[status].reset_index(drop=True)
        metrics_dict['f1_' + name + '_' + status] = x

# plot
plt.figure(facecolor=background_color)
plt.gca().set_facecolor(background_color)


plt.plot(pixel_acc['TrainingAccuracy']*100, color=train_color, label='Train Pixel')
plt.plot(metrics_dict['f1_train_healthy']*100, color=train_color, label='Train Healthy', linestyle='dashed', alpha=0.6, linewidth=0.65)
plt.plot(metrics_dict['f1_train_infested']*100, color=train_color, label='Train Infested', linestyle='dashdot', alpha=0.6, linewidth=0.65)
plt.plot(metrics_dict['f1_train_dead']*100, color=train_color, label='Train Dead', linestyle='dotted', alpha=0.6, linewidth=0.65)

plt.plot(pixel_acc['ValidationAccuracy']*100, color=validation_color, label='Validation Pixel')
plt.plot(metrics_dict['f1_val_healthy']*100, color=validation_color, label='Validation Healthy', linestyle='dashed', alpha=0.6, linewidth=0.65)
plt.plot(metrics_dict['f1_val_infested']*100, color=validation_color, label='Validation Infested', linestyle='dashdot', alpha=0.6, linewidth=0.65)
plt.plot(metrics_dict['f1_val_dead']*100, color=validation_color, label='Validation Dead', linestyle='dotted', alpha=0.6, linewidth=0.65)

plt.xlabel('Epoch')
plt.ylabel('Accuracy / F1 Score')
plt.ylim(0, 100)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(facecolor=background_color, loc='upper left')

plt.savefig((path+'accuracy_plot.png'), bbox_inches='tight', dpi=500)

# Loss Plot
plt.figure(facecolor=background_color)
plt.gca().set_facecolor(background_color)
plt.plot(loss_df['TrainingLoss'], color=train_color, label='Train Loss')
plt.plot(loss_df['ValidationLoss'], color=validation_color, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(facecolor=background_color, loc='upper left')
plt.savefig((path+'loss_plot.png'), bbox_inches='tight', dpi=500)

print(f"""Done""")