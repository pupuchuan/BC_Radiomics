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

    dataset_name = "Dataset003_BC"          # DatasetXXX_NAME的格式
    downloaded_data_dir = "/home/lcp/data/Body_composition/20230601/for_body_composition_seg"     # 原始数据
    nnUNet_raw_data = '/home/lcp/data/Body_composition/nnunet/nnunet_raw_data'  # .../nnUNet_raw_data
    target_base = join(nnUNet_raw_data, dataset_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)

    name_list = os.listdir(join(downloaded_data_dir, 'images'))  # 数据
    # name_list.sort(key=lambda x: int(x[:-7]))   # 排序
    patient_names = []
    test_patient_names = []         # 这里因为没有多余的测试数据，所以先随便放3个数据进去，方便后期做调试，有测试数据的话直接放测试数据就可以了。
    for i, p in enumerate(name_list):
        p_name = p[:-7]
        patdir = join(downloaded_data_dir, p)
        patient_name = "BC_" + p_name
        patient_names.append(patient_name)
        CT = join(downloaded_data_dir, 'images', p)
        seg = join(downloaded_data_dir, 'labels', p)

        shutil.copy(CT, join(target_imagesTr, patient_name + "_0000.nii.gz"))     # _0000是与模态相关的参数，若有多个模态则改为_0001(与后方的channel_name对应)

        shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))

        if i < 3:  # 这里因为没有多余的测试数据，所以先随便放3个数据进去，有测试数据的话直接放测试数据就可以了。
            test_patient_names.append(patient_name)
            shutil.copy(CT, join(target_imagesTs, patient_name + "_0000.nii.gz"))


    generate_dataset_json(
        str(target_base),
        channel_names={
            0: "CT",
        },
        labels={
            "background": 0,
            "Muscle": 1,
            "IMAT": 2,
            "VAT": 3,
            "SAT": 4,
            "Bone": 5,
        },
        file_ending=".nii.gz",
        num_training_cases=len(name_list),
    )
