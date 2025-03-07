"""
author: Li Chuanpu
date: 2023_05_23 21:40
"""

# !/usr/bin/env python
# coding=utf-8

import os
from spine_utils import *
from radiomics_utils import *
import nibabel as nib
import pandas as pd
from metrics import *
import re
from radiomics import featureextractor
import SimpleITK as sitk
import shutil

params = './radio_3D.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(params)

data_dir = r'F:\data\body_composition\20230601\3_lung\GZC5'
img_dir = os.path.join(data_dir, 'nii')
spine_seg_dir = os.path.join(data_dir, 'bone_seg')
composition_seg_dir = os.path.join(data_dir, 'body_composition_seg')
output_metric_dir = os.path.join(data_dir, 'radiomics_3D')
os.makedirs(output_metric_dir, exist_ok=True)

error_message_file = os.path.join(data_dir, 'error_radiomics.xlsx')
error_file_list = []
error_message_list = []

start_level = 'T1'
end_level = 'T12'

spine_name_list = ['S', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                   'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']
spine_label_list = [i for i in range(1, 26)]
spine_dict = {}
for i, name in enumerate(spine_name_list):
    spine_dict[name] = spine_label_list[i]

body_composition_list = ['Muscle', 'IMAT', 'VAT', 'SAT', 'Bone']
body_composition_label = [i for i in range(1, 6)]   # 1-5
composition_dict = {}
for i, name in enumerate(body_composition_list):
    composition_dict[name] = body_composition_label[i]

for file in os.listdir(img_dir):

    print(
        f'---------------------------------------------------{file}--------------------------------------------------------')

    this_output_metric_dir = os.path.join(output_metric_dir, file[:-7].split('_')[0].strip())
    os.makedirs(this_output_metric_dir, exist_ok=True)

    img = nib.load(os.path.join(img_dir, file))

    if len(img.shape) > 3:
        print(f"WARNING: Input image has {len(img.shape)} dimensions. Only using first three dimensions.")
        img = nib.Nifti1Image(img.get_fdata()[:, :, :, 0], img.affine)

    spine_seg = nib.load(os.path.join(spine_seg_dir, file[:-7] + '.nii.gz'))
    composition_seg = nib.load(os.path.join(composition_seg_dir, file[:-7] + '.nii.gz'))

    img, img_spacing = ToCanonical(img)
    spine_seg, spine_spacing = ToCanonical(spine_seg)
    composition_seg, composition_spacing = ToCanonical(composition_seg)

    try:
        centroids_3d, updown_positions = get_position(
            spine_seg,
            img,
            spine_dict,
            connectivity=4,
            s_choice='max'
        )
    except Exception:
        logging.warning(f'!!!!!!!!!!!!!!!!!!!spine seg output error!!!!!!!!!!!!!!!!!!!!!!!!!')
        error_file_list.append(file)
        error_message_list.append('spine_seg error')
        continue

    try:
        start_slice = updown_positions[start_level][-1]
        end_slice = updown_positions[end_level][0]

        assert start_slice > end_slice, 'start and end position may be reversed!'

        img_one = img.get_fdata()[:, :, end_slice: start_slice + 1]
        seg_one = composition_seg.get_fdata()[:, :, end_slice: start_slice + 1]

        total_results_3d = {}
        save_name = os.path.join(this_output_metric_dir, f'{start_level}_{end_level}_radiomics.csv')

        for key, value in composition_dict.items():

            seg_one_whole = np.zeros(
                shape=(img_one.shape[0], img_one.shape[1], img_one.shape[2]))

            seg_one_whole[seg_one == value] = 1

            img_sitk = sitk.GetImageFromArray(img_one.transpose(2, 1, 0))     # nib: x, y, z ; sitk: z,y,x
            img_sitk.SetSpacing(tuple(np.array(img_spacing, dtype=np.float64)))   # [x,y,z]
            seg_itk = sitk.GetImageFromArray(seg_one_whole.transpose(2, 1, 0))
            seg_itk.SetSpacing(tuple(np.array(composition_spacing, dtype=np.float64)))

            result = extractor.execute(img_sitk, seg_itk)
            total_results_3d[key] = result

            save_ra_results(composition_dict, total_results_3d, save_name)
    except Exception:
        logging.warning(
            f'!!!!!!!!!!!!!!!!!!!This file may not include：{start_level}-{end_level}，please check!!!!!!!!!!!!!!!!!!!!!!!!!')
        error_file_list.append(file)
        error_message_list.append(f'This file may not include {start_level}-{end_level}')

df = pd.DataFrame({'error_file': error_file_list, 'error_message': error_message_list})
df.to_excel(error_message_file, sheet_name='sheet1', index=False)