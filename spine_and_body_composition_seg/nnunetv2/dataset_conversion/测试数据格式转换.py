"""
author: Li Chuanpu
date: 2023_05_17 11:21
"""

import os
import shutil

data_dir = '/home/lcp/data/Body_composition/2raw/breast/KMFY/nii'
save_dir = '/home/lcp/data/Body_composition/2raw/breast/KMFY/nnunet_images'
os.makedirs(save_dir, exist_ok=True)

# for patient in os.listdir(data_dir):
#     print(patient)
#     shutil.copy(os.path.join(data_dir, patient), os.path.join(save_dir, 'BC_' + patient.split('.')[0] + '_0000.nii.gz'))

for patient in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, patient)):
        print(file)
        shutil.copy(os.path.join(data_dir, patient, file),
                    os.path.join(save_dir, 'BC_' + file.split('.')[0] + '_0000.nii.gz'))
