"""Module to split a large image with bounding boxes into smaller sub-
samples.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from PIL import Image
import rasterio

from utils import calculate_area

pd.options.mode.chained_assignment = None


######## CONFIG ########
location_id = 'Hachenburg_loc2_23-05-26'
image_path = '../Data/OrthoImages/' + location_id + '/odm_orthophoto/odm_orthophoto.tif'
labels_path = '../Data/OrthoImages/' + location_id + '/labels.csv'
image_output_folder = '../Data/ProcessedImages/'
csv_output_folder = '../Data/ProcessedImagesCSVs/'
out_image_size = 512 # optional, leave default value in doubt
verbose = True # optional, leave default value in doubt
visual_sanity_check = True # optional, leave default value in doubt
min_show_tree = 2 # optional, leave default value in doubt
######## CONFIG ########

if __name__ == '__main__':

    # Read the .tif file using rasterio
    with rasterio.open(image_path) as ds:
        # Read the image data
        image_data = ds.read()
        # Transpose the data to fit imshow
        image_data = np.transpose(image_data, (1, 2, 0))
        image_data = image_data[:, :, [0, 1, 2]]

    # read in label data
    label_data = pd.read_csv(labels_path)

    # extract bounding numbers for the image
    end_num_1 = int(np.floor(image_data.shape[0] / out_image_size))
    end_num_2 = int(np.floor(image_data.shape[1] / out_image_size))

    # initialize loop objects
    counter = 1
    bboxes_list = []

    # got through the sub-images given the specified output size and cut out
    # smaller images and bounding boxes
    for i in range(end_num_1):
        for j in range(end_num_2):
            
            # next image region
            crop_image = image_data[512*i:512*(i+1), 512*j:512*(j+1), [0, 1, 2]]

            # extract pixel range to look for bounding boxes
            x_range = (512*j, 512*(j+1))
            y_range = (512*i, 512*(i+1))

            # get pixel in image range
            crop_pixels = label_data[
                ((label_data['x_min'].between(x_range[0], x_range[1])) |
                (label_data['x_max'].between(x_range[0], x_range[1]))) &
                ((label_data['y_min'].between(y_range[0], y_range[1])) |
                (label_data['y_max'].between(y_range[0], y_range[1])))
            ].copy()

            # if there are no bboxes/trees in the image region jump to next image
            if not crop_pixels.shape[0] > 0:
                continue
                
            # adjust bounding boxes to new image pixels
            crop_pixels['x_min'] = crop_pixels['x_min'] - x_range[0]
            crop_pixels['x_max'] = crop_pixels['x_max'] - x_range[0]
            crop_pixels['y_min'] = crop_pixels['y_min'] - y_range[0]
            crop_pixels['y_max'] = crop_pixels['y_max'] - y_range[0]

            # set bounding boxes to upper/lower size
            pixel_columns = ['x_min', 'x_max', 'y_min', 'y_max']
            crop_pixels[pixel_columns] = crop_pixels[pixel_columns].clip(lower=0)
            crop_pixels[pixel_columns] = crop_pixels[pixel_columns] \
                .clip(upper=out_image_size)

            # check areas of boxes
            crop_pixels['area'] = crop_pixels.apply(
                lambda row: calculate_area(
                    row['x_min'], row['x_max'], row['y_min'], row['y_max']
                ), axis=1
            )

            # remove boxes that are only part-wise present
            crop_pixels = crop_pixels.query('area >= 500')

            # if there were only part of boxes in image jump to next loop
            if not crop_pixels.shape[0] > 0:
                continue

            # specify name of the new image
            img_name = location_id + '_img_' + '{:04n}'.format(counter) + '.png'

            # append image name, bounding boxes and labels
            for _, row in crop_pixels.iterrows():
                bboxes_list.append([
                    img_name,
                    row['x_min'], row['x_max'], row['y_min'], row['y_max'],
                    row['class_label'], row['class_no']
                ])
            
            # store image
            Image.fromarray(crop_image).save(image_output_folder + img_name)

            if verbose:
                print(
                    f"{img_name} saved with {crop_pixels.shape[0]} bounding boxes"
                )

            # increase counter
            counter += 1

    # process label dataframe
    bbox_df = pd.DataFrame(
        bboxes_list,
        columns=['image', 'x_min', 'x_max', 'y_min', 'y_max',
                 'class_label', 'class_no']
    )

    # show how many trees are present in the new dataset
    print(bbox_df['class_label'].value_counts())

    # save labels as csv
    csv_name = csv_output_folder + 'labels_' + location_id + '.csv'
    bbox_df.to_csv(csv_name, index=False)

    print(f"{csv_name} saved.")




    # display example image as visual sanity check
    if visual_sanity_check:

        # pick image with min. 2 trees
        display_img = random.choice(
            bbox_df.groupby('image').count() \
                .query('class_label >= @min_show_tree') \
                .reset_index() \
                ['image']
        )

        # read in image
        image = plt.imread(image_output_folder + display_img)

        # print bounding box values
        print(bbox_df.query('image == @display_img'))

        # specify color dict
        col = {
            'dead': 'black',
            'healthy': 'blue',
            'infested': 'red'
        }

        # make plot
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # add bounding boxes
        for _, row in bbox_df.query('image == @display_img').iterrows():
            
            # make box
            box = patches.Rectangle(
                (row['x_min'], row['y_min']),
                row['x_max']-row['x_min'], row['y_max']-row['y_min'],
                linewidth=1, edgecolor=col.get(row['class_label']),
                facecolor='none'
            )
            ax.add_patch(box)

        print('Done ✓')
        
        # show image
        plt.show()
