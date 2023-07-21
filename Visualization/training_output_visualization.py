"""Script for Visualizing the Accuracy Metrics
during model training.
"""

import matplotlib.pyplot as plt
import pandas as pd

######## CONFIG ########
output_folder = '20230720_0721'
show_plot = False
save_plot = True
background_color = '#eff1f3'
train_color = '#222843'
validation_color = '#dda15e'
######## CONFIG ########

# generate path
path = '../../DriveCompatibleCode/Detection/Output/' + output_folder + '/'

# read in data
loss_df = pd.read_csv(path+'loss_df.csv')
training_map = pd.read_csv(path+'training_MAP.csv')
validation_map = pd.read_csv(path+'validation_MAP.csv')

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
if save_plot:
    plt.savefig((path+'loss_plot.png'), bbox_inches='tight', dpi=500)
if show_plot:
    plt.show()

# Accuracy Plot
# Loss Figure
plt.figure(facecolor=background_color)
plt.gca().set_facecolor(background_color)
plt.plot(training_map['map']*100, color=train_color, label='Train mAP')
plt.plot(training_map['map_50']*100, color=train_color, label='Train mAP50', 
         linestyle='dashed', alpha=0.6, linewidth=0.65)
plt.plot(training_map['mar_100']*100, color=train_color, label='Train mAR@100',
         linestyle='dotted', alpha=0.6, linewidth=0.65)
plt.plot(validation_map['map']*100, color=validation_color, label='Validation mAP')
plt.plot(validation_map['map_50']*100, color=validation_color, label='Validation mAP50',
         linestyle='dashed', alpha=0.6, linewidth=0.65)
plt.plot(validation_map['mar_100']*100, color=validation_color, label='Validation mAR@100',
         linestyle='dotted', alpha=0.6, linewidth=0.65)
plt.xlabel('Epoch')
plt.ylabel('mAP / mAR (in %)')
plt.ylim(0, 100)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(facecolor=background_color, loc='upper left')
if save_plot:
    plt.savefig((path+'accuracy_plot.png'), bbox_inches='tight', dpi=500)
if show_plot:
    plt.show()

print(f"""Done ✓""")
