"""
"""
print(f"""Load Modules and Configuration. This might take a while.""")

import torch
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import json

from Detection.utils import extract_bboxes_from_xml, get_adjusted_predictions, \
    get_adjusted_ground_truth, precision_recall_f1score_detection, \
    plot_pred_vs_true

inference_detection_model_list = pd.read_csv('./inference_detection_model_list.csv', delimiter=';')

runs = os.listdir('./Output')
runs = [r for r in runs if 'v2' in r]

for run in runs:
    
    print(run)
    
    f = os.listdir('Output/' + run + '/')
    if 'Inference' in f:
        f2 = os.listdir('Output/' + run + '/' + 'Inference')
        if 'overall_result_df.csv' in f2:
            print('Skip')
            continue

    model_oi = [m for m in inference_detection_model_list['model'].unique() if m in run][0]
    location_oi = [l for l in inference_detection_model_list['location_id'].unique() if l in run][0]
    model = inference_detection_model_list.query('location_id == @location_oi & model == @model_oi')['epoch'].item()
    model = 'epoch_' + str(model) + '_model.pth'


    ######## CONFIG ########
    test_location_id = location_oi # 'Haiterbach_loc1'
    run = run + '/' # '20230824_0916_test_all_FRCNN_resnet/'
    model = model # 'epoch_48_model.pth'
    all = False # all images and test images used
    run_path = 'Output/' + run
    save_path = run_path + 'Inference/'
    img_save_path = save_path + 'PredictionImages/'
    image_path = 'Data/ProcessedImages/'
    bbox_path = 'Data/ProcessedImages/'
    #image_path = 'C:/Users/leona/Documents/06_MasterThesis/Data/BackUp/Backup_2023-11-23/ProcessedImages/'
    #bbox_path = 'C:/Users/leona/Documents/06_MasterThesis/Data/BackUp/Backup_2023-11-23/ProcessedImages/'
    img_format = '.png'
    bbox_format = '.xml'
    classes = ['healthy', 'infested', 'dead']
    threshold_dict = {'healthy': 0.5, 'infested': 0.1, 'dead': 0.5}
    ######## CONFIG ########
    


    # read in model
    model_path = os.path.join(run_path, 'models', model)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    print(f"""Model loaded ✓""")

    # extract image files
    if all:
        print(f"""Test Location: All""")
        with open('./Data/test_images_v2.json') as f:
            img_files = json.load(f)
            img_files = [i[:-4] for i in img_files]
    else:
        print(f"""Test Location: {test_location_id} ({run_path})""")
        img_files = [i[:-4] for i in os.listdir(image_path) if test_location_id in i]

    # prepare storage dict
    result_dict = {'Image': []}
    for c in classes:
        result_dict[f'tp_{c}'] = []
        result_dict[f'fp_{c}'] = []
        result_dict[f'fn_{c}'] = []
        result_dict[f'precision_{c}'] = []
        result_dict[f'recall_{c}'] = []
        result_dict[f'f1_score_{c}'] = []

    # make output paths
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + 'PredictionImages/'):
        os.makedirs(save_path + 'PredictionImages/')

    print(f"""Start Prediction evaluation for {len(img_files)} test images""")

    # loop through images and evaluate
    for img_file in tqdm(img_files):
        
        # read in image
        img_path = image_path + img_file + '.png'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = [torch.as_tensor(image/255.).permute([2, 0, 1])]

        # extract ground truth bboxes
        xml_path = bbox_path + img_file + '.xml'
        ground_truth = extract_bboxes_from_xml(bboxes_path=xml_path, class_label_name='name')
        ground_truth = list(zip(ground_truth[0], ground_truth[1]))
        
        result_dict['Image'].append(img_file)

        # make predictions
        with torch.no_grad():
            predictions = model(image)

        # adjust predictions and ground truth
        pred_dict, pred_all_boxes = get_adjusted_predictions(predictions, threshold_dict=threshold_dict)
        gt_dict = get_adjusted_ground_truth(ground_truth)

        # evaluate prediction
        for c in classes:
            metrics_dict = precision_recall_f1score_detection(
                pred_boxes=pred_dict[c]['boxes'],
                true_boxes=gt_dict[c]['boxes'],
                iou_threshold=0.5
            )
            result_dict[f'tp_{c}'].append(metrics_dict['tp'])
            result_dict[f'fp_{c}'].append(metrics_dict['fp'])
            result_dict[f'fn_{c}'].append(metrics_dict['fn'])
            result_dict[f'precision_{c}'].append(metrics_dict['precision'])
            result_dict[f'recall_{c}'].append(metrics_dict['recall'])
            result_dict[f'f1_score_{c}'].append(metrics_dict['f1_score'])

        # save prediction visuals
        plot_pred_vs_true(image_path=img_path,
                        all_boxes_pred=pred_all_boxes,
                        all_boxes_ground_truth=ground_truth,
                        save_path=img_save_path + img_file + 'predictions.png',
                        show=False)

        # save result frame
        result_df = pd.DataFrame(result_dict)
        result_df.to_csv(save_path + 'result_df.csv', index=False)

    # calculate and save overall results
    metrics = ['n', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1_score']
    overall_res_dict = {}
    for m in metrics:
        overall_res_dict[m] = []
    for c in classes:
        tp = np.sum(result_df[f'tp_{c}'])
        fp = np.sum(result_df[f'fp_{c}'])
        fn = np.sum(result_df[f'fn_{c}'])
        n = tp + fn
        precision = (tp + fp) and tp / (tp + fp) or 0
        recall = (tp + fn) and tp / (tp + fn) or 0
        f1_score = (tp + fp + fn) and 2*tp / (2*tp + fp + fn) or 0
        overall_res_dict['n'].append(n)
        overall_res_dict['tp'].append(tp)
        overall_res_dict['fp'].append(fp)
        overall_res_dict['fn'].append(fn)
        overall_res_dict['precision'].append(round(precision, 4))
        overall_res_dict['recall'].append(round(recall, 4))
        overall_res_dict['f1_score'].append(round(f1_score, 4))
    overall_res_df = pd.DataFrame.from_dict(overall_res_dict, orient='index')
    overall_res_df.columns = classes
    overall_res_df = overall_res_df.rename_axis('metric').reset_index()
    overall_res_df.to_csv(save_path + 'overall_result_df.csv', index=False)

    print(f"""Prediction evaluation for {len(img_files)} test images done ✓""")
