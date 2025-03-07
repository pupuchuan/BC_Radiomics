import os
import shutil
from pathlib import Path

import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import shutil

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def make_out_dirs(dataset_id: int, task_name="ACDC"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir


def copy_files(src_data_folder: Path, train_dir: Path, labels_dir: Path, test_dir: Path):
    """Copy files from the ACDC dataset to the nnUNet dataset folder. Returns the number of training cases."""
    patients_train = sorted([f for f in (src_data_folder / "training").iterdir() if f.is_dir()])
    patients_test = sorted([f for f in (src_data_folder / "testing").iterdir() if f.is_dir()])

    num_training_cases = 0
    # Copy training files and corresponding labels.
    for patient_dir in patients_train:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                # The stem is 'patient.nii', and the suffix is '.gz'.
                # We split the stem and append _0000 to the patient part.
                shutil.copy(file, train_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")
                num_training_cases += 1
            elif file.suffix == ".gz" and "_gt" in file.name:
                shutil.copy(file, labels_dir / file.name.replace("_gt", ""))

    # Copy test files.
    for patient_dir in patients_test:
        for file in patient_dir.iterdir():
            if file.suffix == ".gz" and "_gt" not in file.name and "_4d" not in file.name:
                shutil.copy(file, test_dir / f"{file.stem.split('.')[0]}_0000.nii.gz")

    return num_training_cases


def convert_acdc(src_data_folder: str, dataset_id=27):
    out_dir, train_dir, labels_dir, test_dir = make_out_dirs(dataset_id=dataset_id)
    num_training_cases = copy_files(Path(src_data_folder), train_dir, labels_dir, test_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={
            "background": 0,
            "RV": 1,
            "MLV": 2,
            "LVC": 3,
        },
        file_ending=".nii.gz",
        num_training_cases=num_training_cases,
    )


if __name__ == "__main__":

    dataset_name = "Dataset004_SpineSeg"          # DatasetXXX_NAME的格式
    downloaded_data_dir = "/home/lcp/data/Body_composition/spine_train"     # 原始数据
    nnUNet_raw_data = '/home/lcp/data/Body_composition/nnunet/nnunet_raw_data'  # .../nnUNet_raw_data
    target_base = join(nnUNet_raw_data, dataset_name)
    target_imagesTr = join(target_base, "imagesTr")   # 训练图像
    target_labelsTr = join(target_base, "labelsTr")   # 训练标签
    target_imagesTs = join(target_base, "imagesTs")   # 测试图像

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)

    count = 0
    for date in os.listdir(downloaded_data_dir):
        for hospital in os.listdir(join(downloaded_data_dir, date)):
            for patient_file in os.listdir(join(downloaded_data_dir, date, hospital, 'nii')):
                if date == '3_abdomen':
                    p_name = patient_file[:-12]
                else:
                    p_name = patient_file[:-7]

                CT = join(downloaded_data_dir,date, hospital, 'nii', patient_file)
                seg = join(downloaded_data_dir,date, hospital, 'bone_seg', p_name + '.nii.gz')

                final_output_name = "SP_" + date + '-' + hospital + '-' + p_name

                shutil.copy(seg, join(target_labelsTr, final_output_name + ".nii.gz"))
                shutil.copy(CT, join(target_imagesTr, final_output_name + "_0000.nii.gz"))     # _0000是与模态相关的参数，若有多个模态则改为_0001(与后方的channel_name对应)

                count += 1


    # name_list = os.listdir(join(downloaded_data_dir, 'images'))  # 数据
    # # name_list.sort(key=lambda x: int(x[:-7]))   # 排序
    # patient_names = []
    # test_patient_names = []         # 这里因为没有多余的测试数据，所以先随便放3个数据进去，方便后期做调试，有测试数据的话直接放测试数据就可以了。
    # for i, p in enumerate(name_list):
    #     p_name = p[:-7]
    #     patdir = join(downloaded_data_dir, p)
    #     patient_name = "SP_" + p_name
    #     patient_names.append(patient_name)
    #     CT = join(downloaded_data_dir, 'images', p)
    #     seg = join(downloaded_data_dir, 'labels', p)
    #
    #     try:
    #         shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))
    #         shutil.copy(CT, join(target_imagesTr, patient_name + "_0000.nii.gz"))     # _0000是与模态相关的参数，若有多个模态则改为_0001(与后方的channel_name对应)
    #     except Exception:
    #         # shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))
    #         shutil.copy(CT, join(target_imagesTs, patient_name + "_0000.nii.gz"))  # _0000是与模态相关的参数，若有多个模态则改为_0001(与后方的channel_name对应)


        # if i < 3:  # 这里因为没有多余的测试数据，所以先随便放3个数据进去，有测试数据的话直接放测试数据就可以了。
        #     test_patient_names.append(patient_name)
        #     shutil.copy(CT, join(target_imagesTs, patient_name + "_0000.nii.gz"))

    # 初始化字典和值
    spine_name_list = ['background', 'S', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                       'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']  # 注意：数据里有些可能没有
    modi_spine_dict = {}
    modi_spine_label_list = [i for i in range(0, 26)]  # 1-25 spine是1, 以此类推
    for i, name in enumerate(spine_name_list):
        modi_spine_dict[name] = modi_spine_label_list[i]

    generate_dataset_json(
        str(target_base),
        channel_names={
            0: "CT",
        },
        labels=modi_spine_dict,
        file_ending=".nii.gz",
        num_training_cases=count,
    )
