"""
author: Li Chuanpu
date: 2023_06_02 16:37
"""

from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import *
import math
import random

random.seed(0)

dataset_name = 'Dataset001_BodyComposition'
data_dir = os.path.join(nnUNet_raw, dataset_name, 'labelsTr')

# custom split to ensure we are stratifying properly. This dataset only has 1 folds
caseids = os.listdir(data_dir)
combine_type = {}      # 计算一下 部位 + 医院对应的数量，用来分折
for case in caseids:
    one_type = case.split('-')[0] + '-' + case.split('-')[1]
    if one_type in combine_type:
        combine_type[one_type].append(case)
    else:
        combine_type[one_type] = []
        combine_type[one_type].append(case)

train_list = []
val_list = []

for keys, values in combine_type.items():
    if keys == 'SP_whole-GDSY':      # 这个勾错的体数据作为验证集
        val_list.append('SP_whole-GDSY-10334320')
        values.remove('SP_whole-GDSY-10334320.nii.gz')
        for sample in values:
            train_list.append(sample.replace('.nii.gz', ''))
    else:
        val_num = round(len(values) / 10)
        val_sample = random.sample(values, val_num)
        train_sample = [x for x in values if x not in val_sample]
        for sample in val_sample:
            val_list.append(sample.replace('.nii.gz', ''))
        for sample in train_sample:
            train_list.append(sample.replace('.nii.gz', ''))

# 只跑一折
splits = [{'train':train_list, 'val': val_list}]
save_json(splits, join(nnUNet_preprocessed, dataset_name, 'splits_final1.json'))