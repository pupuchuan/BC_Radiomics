"""
author: Li Chuanpu
date: 2023_07_14 18:15
"""
import os

data_dir = '/home/lcp/data/QCT/nii'

for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    new_file_path = file_path[:-7] + '_0000.nii.gz'
    os.rename(file_path, new_file_path)