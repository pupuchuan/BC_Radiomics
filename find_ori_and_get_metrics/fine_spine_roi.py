"""
author: Li Chuanpu
date: 2023_05_23 21:40
"""

import os
from spine_utils import *
import nibabel as nib

# 初始化字典和值
spine_name_list = ['S', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                   'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']    # 注意：数据里有些可能没有
spine_label_list = [i for i in range(18, 42)]
spine_label_list.insert(0, 92)   # 骶骨标签
spine_dict = {}
for i, name in enumerate(spine_name_list):
    spine_dict[name] = spine_label_list[i]

# data_dir = '/home/lcp/data/Body_composition/2raw/breast'
# # for hospital in os.listdir(data_dir):
# for hospital in ['GDSY', 'TCIA']:
#     os.makedirs(os.path.join(data_dir, hospital, 'bone_roi_4'), exist_ok=True)
#     for nii in os.listdir(os.path.join(data_dir, hospital, 'nii')):
#         print(f'---------------------------------------------------{nii}--------------------------------------------------------')
#         output_dir = os.path.join(data_dir, hospital, 'bone_roi_4', nii[:-7])
#         os.makedirs(output_dir, exist_ok=True)
#         img = nib.load(os.path.join(data_dir, hospital, 'nii', nii))
#         if len(img.shape) > 3:
#             print(f"WARNING: Input image has {len(img.shape)} dimensions. Only using first three dimensions.")
#             img = nib.Nifti1Image(img.get_fdata()[:, :, :, 0], img.affine)
#
#         seg = nib.load(os.path.join(data_dir, hospital, 'bone_seg', nii))
#
#         # 重置一下轴
#         img, img_spacing = ToCanonical(img)
#         seg, seg_spacing = ToCanonical(seg)
#
#         # 计算ROI的位置
#         (spine_hus, rois, centroids_3d, updown_positions) = compute_rois(
#             seg,  # 脊柱分割结果 (nib)
#             img,  # 原始图像 (nib)
#             spine_dict,  # 传入一个categories的dict: 表示每个椎骨对应的值
#         )
#
#         # 可视化
#         img_sagittal, img_coronal = visualize_coronal_sagittal_spine(
#             seg.get_fdata(),
#             list(rois.values()),
#             img.get_fdata(),
#             centroids_3d,
#             updown_positions,
#             output_dir,
#             spine_hus=spine_hus,
#             model_type=spine_dict,
#             pixel_spacing=img_spacing,
#             format='png',
#         )

data_dir = r'/home/lcp/data/Body_composition/2raw/breast/KMFY/nii'
for patient in os.listdir(data_dir):
    for nii in os.listdir(os.path.join(data_dir, patient)):
        print(
        f'---------------------------------------------------{nii}--------------------------------------------------------')
        output_dir = os.path.join(data_dir[:-4], 'bone_roi_4', nii[:-7])
        os.makedirs(output_dir, exist_ok=True)
        img = nib.load(os.path.join(data_dir, patient, nii))
        if len(img.shape) > 3:
            print(f"WARNING: Input image has {len(img.shape)} dimensions. Only using first three dimensions.")
            img = nib.Nifti1Image(img.get_fdata()[:, :, :, 0], img.affine)

        seg = nib.load(os.path.join(data_dir[:-4], 'bone_seg', nii))

        # 重置一下轴
        img, img_spacing = ToCanonical(img)
        seg, seg_spacing = ToCanonical(seg)

        # 计算ROI的位置
        (spine_hus, rois, centroids_3d, updown_positions) = compute_rois(
            seg,  # 脊柱分割结果 (nib)
            img,  # 原始图像 (nib)
            spine_dict,  # 传入一个categories的dict: 表示每个椎骨对应的值
        )

        # 可视化
        img_sagittal, img_coronal = visualize_coronal_sagittal_spine(
            seg.get_fdata(),
            list(rois.values()),
            img.get_fdata(),
            centroids_3d,
            updown_positions,
            output_dir,
            spine_hus=spine_hus,
            model_type=spine_dict,
            pixel_spacing=img_spacing,
            format='png',
        )

        

# data_dir = r'/home/lcp/data/Body_composition/1raw/nii'
# for nii in os.listdir(data_dir):
#     print(f'---------------------------------------------------{nii}--------------------------------------------------------')
#     output_dir = os.path.join(data_dir[:-4], 'bone_roi_true', nii[:-7])
#     os.makedirs(output_dir, exist_ok=True)
#     img = nib.load(os.path.join(data_dir, nii))
#     if len(img.shape) > 3:
#         print(f"WARNING: Input image has {len(img.shape)} dimensions. Only using first three dimensions.")
#         img = nib.Nifti1Image(img.get_fdata()[:, :, :, 0], img.affine)
#
#     seg = nib.load(os.path.join(data_dir[:-4], 'bone_seg', nii))
#
#     # 重置一下轴
#     img, img_spacing = ToCanonical(img)
#     seg, seg_spacing = ToCanonical(seg)
#
#     # 计算ROI的位置 （中心点）
#     (spine_hus, rois, centroids_3d, updown_positions) = compute_rois(
#         seg,  # 脊柱分割结果 (nib)
#         img,  # 原始图像 (nib)
#         spine_dict,  # 传入一个categories的dict: 表示每个椎骨对应的值
#     )
#
#     # 可视化
#     img_sagittal, img_coronal = visualize_coronal_sagittal_spine(
#         seg.get_fdata(),
#         list(rois.values()),
#         img.get_fdata(),
#         centroids_3d,
#         updown_positions,
#         output_dir,
#         spine_hus=spine_hus,
#         model_type=spine_dict,
#         pixel_spacing=img_spacing,
#         format='png',
#     )









