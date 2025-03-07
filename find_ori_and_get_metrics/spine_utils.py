import logging
import math
from glob import glob
from typing import Dict, List

import cv2
import numpy as np
from pydicom.filereader import dcmread
from scipy.ndimage import zoom
import nibabel as nib
from scipy.ndimage.measurements import center_of_mass

import spine_visualization

spine_name_list = ['S', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                   'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']

def ToCanonical(img):
    """
    First dim goes from L to R.
    Second dim goes from P to A.
    Third dim goes from I to S.
    """
    img = nib.as_closest_canonical(img)
    pixel_spacing_list = img.header.get_zooms()

    return img, pixel_spacing_list

def find_spine_dicoms(centroids: Dict, path: str, levels):
    """Find the dicom files corresponding to the spine T12 - L5 levels."""

    vertical_positions = []
    for level in centroids:
        centroid = centroids[level]
        vertical_positions.append(round(centroid[2]))

    dicom_files = []
    ipps = []
    for dicom_path in glob(path + "/*.dcm"):
        ipp = dcmread(dicom_path).ImagePositionPatient
        ipps.append(ipp[2])
        dicom_files.append(dicom_path)

    dicom_files = [x for _, x in sorted(zip(ipps, dicom_files))]
    dicom_files = list(np.array(dicom_files)[vertical_positions])

    return (dicom_files, levels, vertical_positions)


# Function that takes a numpy array as input, computes the
# sagittal centroid of each label and returns a list of the
# centroids
def compute_centroids(seg: np.ndarray, spine_model_type, ori="sagittal"):
    """Compute the centroids of the labels.

    Args:
        seg (np.ndarray): Segmentation volume.
        spine_model_type (str): Model type.

    Returns:
        List[int]: List of centroids.
    """
    # take values of spine_model_type.categories dictionary
    # and convert to list
    centroids = {}
    for level in spine_model_type:
        label_idx = spine_model_type[level]
        try:
            pos = compute_centroid(seg, ori, label_idx)
            centroids[level] = pos
        except Exception:
            logging.warning(f"Label {level} not found in segmentation volume.")
    return centroids


# Function that takes a numpy array as input, as well as a list of centroids,
# takes a slice through the centroid on axis = 1 for each centroid
# and returns a list of the slices
def get_slices(seg: np.ndarray, centroids: Dict, spine_model_type, ori="sagittal"):
    """Get the slices corresponding to the centroids.

    Args:
        seg (np.ndarray): Segmentation volume.
        centroids (List[int]): List of centroids.
        spine_model_type (str): Model type.

    Returns:
        List[np.ndarray]: List of slices.
    """
    seg = seg.astype(np.uint8)
    slices = {}
    for level in centroids:
        label_idx = spine_model_type[level]
        if ori == 'sagittal':
            binary_seg = (seg[centroids[level], :, :] == label_idx).astype(int)
        elif ori == "coronal":
            binary_seg = (seg[:, centroids[level], :] == label_idx).astype(int)
        else:
            binary_seg = (seg[:, :, centroids[level]] == label_idx).astype(int)
        if np.sum(binary_seg) > 50:  # heuristic to make sure enough of the body is showing   这个值不可以太大, 否则厚层的数据会出错！
            slices[level] = binary_seg
        # slices[level] = binary_seg
    return slices


# Function that takes a mask and for each deletes the right most
# connected component. Returns the mask with the right most
# connected component deleted
def delete_right_most_connected_component(mask: np.ndarray):
    """Delete the right most connected component corresponding to spinous processes.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Mask volume.
    """
    mask = mask.astype(np.uint8)
    _, labels, status, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    right_most_connected_component = np.argmin(centroids[1:, 1]) + 1

    mask[labels == right_most_connected_component] = 0
    return mask

def delete_small_connected_component(mask: np.ndarray):
    """Delete the right most connected component corresponding to spinous processes.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Mask volume.
    """
    mask = mask.astype(np.uint8)
    _, labels, status, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    right_most_connected_component = np.argmin(status[1:, -1]) + 1    # 把小的那一块排除
    # 多加一个判断，防止骶骨出错

    mask[labels == right_most_connected_component] = 0
    return mask

# compute center of mass of 2d mask
def compute_center_of_mass(mask: np.ndarray, connectivity: int):
    """Compute the center of mass of a 2D mask.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Center of mass.
    """
    mask = mask.astype(np.uint8)
    _, _, _, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    center_of_mass = np.mean(centroids[1:, :], axis=0)
    return center_of_mass

# compute up position and down position of 2d mask
def compute_up_and_down(mask: np.ndarray):
    """Compute the center of mass of a 2D mask.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Center of mass.
    """
    mask = mask.astype(np.uint8)
    up_position = np.where(mask!=0)[1].min()
    down_position = np.where(mask!=0)[1].max()
    return np.array([up_position, down_position])

def compute_right_and_left(mask: np.ndarray):
    """Compute the center of mass of a 2D mask.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Center of mass.
    """
    mask = mask.astype(np.uint8)
    up_position = np.where(mask!=0)[0].min()
    down_position = np.where(mask!=0)[0].max()
    return np.array([up_position, down_position])

# Function that takes a 3d image and a 3d binary mask and returns that average
# value of the image inside the mask
def mean_img_mask(img: np.ndarray, mask: np.ndarray, index: int):
    """Compute the mean of an image inside a mask.

    Args:
        img (np.ndarray): Image volume.
        mask (np.ndarray): Mask volume.
        rescale_slope (float): Rescale slope.
        rescale_intercept (float): Rescale intercept.

    Returns:
        float: Mean value.
    """
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)
    img_masked = (img * mask)[mask > 0]
    # mean = (rescale_slope * np.mean(img_masked)) + rescale_intercept
    # median = (rescale_slope * np.median(img_masked)) + rescale_intercept
    mean = np.mean(img_masked)
    return mean

def get_position(seg, img, spine_model_type, connectivity, s_choice='normal'):
    """Compute the ROIs for the spine.

    Args:
        seg (np.ndarray): Segmentation volume.
        img (np.ndarray): Image volume.
        rescale_slope (float): Rescale slope.
        rescale_intercept (float): Rescale intercept.
        spine_model_type (Models): Model type.
        connectivity(int): 4 or 8
        s_choice:
    Returns:
        spine_hus (List[float]): List of HU values.
        rois (List[np.ndarray]): List of ROIs.
        centroids_3d (List[np.ndarray]): List of centroids.
    """
    seg_np = seg.get_fdata()
    centroids = compute_centroids(seg_np, spine_model_type)
    slices = get_slices(seg_np, centroids, spine_model_type)
    for i, level in enumerate(slices):
        slice = slices[level]
        # keep only the two largest connected components
        two_largest, two = keep_two_largest_connected_components(slice, connectivity)
        if two:
            if level == 'S' and s_choice == 'max':
                slices[level] = delete_small_connected_component(two_largest)
            else:
                slices[level] = delete_right_most_connected_component(two_largest)

    if (two == False):
            slices.pop(level)


    # Compute ROIs
    # rois = {}
    # spine_hus = {}
    centroids_3d = {}
    updown_positions = {}
    for i, level in enumerate(slices):
        slice = slices[level]
        center_of_mass = compute_center_of_mass(slice, connectivity=connectivity)
        centroid = np.array([centroids[level], center_of_mass[1], center_of_mass[0]])    # xyz
        updown = compute_up_and_down(slice)   #
        try:
            centroids_3d[level] = centroid
            updown_positions[level] = updown
        except Exception:
            logging.warning(f"Label {level} too small in the boundary! delete it.")
    return centroids_3d, updown_positions

def keep_two_largest_connected_components(mask: Dict, connectivity: int):
    """Keep the two largest connected components.

    Args:
        mask (np.ndarray): Mask volume.

    Returns:
        np.ndarray: Mask volume.
    """
    mask = mask.astype(np.uint8)
    # sort connected components by size
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)     # 8连通和4连通？
    stats = stats[1:, 4]
    sorted_indices = np.argsort(stats)[::-1]
    # keep only the two largest connected components
    mask = np.zeros(mask.shape)
    mask[labels == sorted_indices[0] + 1] = 1
    two = True
    try:
        mask[labels == sorted_indices[1] + 1] = 1
    except Exception:
        two = False
    return (mask, two)


def compute_centroid(seg: np.ndarray, plane: str, label: int):
    """Compute the centroid of a label in a given plane.

    Args:
        seg (np.ndarray): Segmentation volume.
        plane (str): Plane.
        label (int): Label.

    Returns:
        int: Centroid.
    """
    if plane == "axial":
        sum_out_axes = (0, 1)
        sum_axis = 2
    elif plane == "sagittal":
        sum_out_axes = (1, 2)
        sum_axis = 0
    elif plane == "coronal":
        sum_out_axes = (0, 2)
        sum_axis = 1
    sums = np.sum(seg == label, axis=sum_out_axes)
    normalized_sums = sums / np.sum(sums)
    pos = int(np.sum(np.arange(0, seg.shape[sum_axis]) * normalized_sums))
    return pos


def to_one_hot(label: np.ndarray, model_type, spine_hus):
    """Convert a label to one-hot encoding.

    Args:
        label (np.ndarray): Label volume.
        model_type (Models): Model type.

    Returns:
        np.ndarray: One-hot encoding volume.
    """
    levels = list(spine_hus.keys())
    levels.reverse()
    one_hot_label = np.zeros((label.shape[0], label.shape[1], len(levels)))
    for i, level in enumerate(levels):
        label_idx = model_type[level]
        one_hot_label[:, :, i] = (label == label_idx).astype(int)
    return one_hot_label


def visualize_coronal_sagittal_spine(
    seg: np.ndarray,
    rois: List[np.ndarray],
    mvs: np.ndarray,
    centroids_3d: dict,
    updown_positions:dict,
    output_dir: str,
    spine_hus=None,
    model_type=None,
    pixel_spacing=None,
    format="png",
):
    """Visualize the coronal and sagittal planes of the spine.

    Args:
        seg (np.ndarray): Segmentation volume.
        rois (List[np.ndarray]): List of ROIs.
        mvs (dm.MedicalVolume): Medical volume.
        centroids (List[int]): List of centroids.
        label_text (List[str]): List of labels.
        output_dir (str): Output directory.
        spine_hus (List[float], optional): List of HU values. Defaults to None.
        model_type (Models, optional): Model type. Defaults to None.
    """
    centroids_3d_val = list(centroids_3d.values())
    sagittal_vals, coronal_vals = curved_planar_reformation(mvs, centroids_3d_val)
    zoom_factor = pixel_spacing[2] / pixel_spacing[1]

    sagittal_image = mvs[sagittal_vals, :, range(len(sagittal_vals))]
    sagittal_label = seg[sagittal_vals, :, range(len(sagittal_vals))]
    sagittal_image = zoom(sagittal_image, (zoom_factor, 1), order=3)
    sagittal_label = zoom(sagittal_label, (zoom_factor, 1), order=1).round()    # 根据spacing进行插值

    one_hot_sag_label = to_one_hot(sagittal_label, model_type, spine_hus)
    for roi in rois:
        one_hot_roi_label = roi[sagittal_vals, :, range(len(sagittal_vals))]
        one_hot_roi_label = zoom(one_hot_roi_label, (zoom_factor, 1), order=1).round()
        one_hot_sag_label = np.concatenate(
            (
                one_hot_sag_label,
                one_hot_roi_label.reshape(
                    (one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1)
                ),
            ),
            axis=2,
        )

    coronal_image = mvs[:, coronal_vals, range(len(coronal_vals))]
    coronal_label = seg[:, coronal_vals, range(len(coronal_vals))]
    coronal_image = zoom(coronal_image, (1, zoom_factor), order=3)
    coronal_label = zoom(coronal_label, (1, zoom_factor), order=1).round()

    # coronal_image = zoom(coronal_image, (zoom_factor, 1), order=3)
    # coronal_label = zoom(coronal_label, (zoom_factor, 1), order=0).astype(int)

    one_hot_cor_label = to_one_hot(coronal_label, model_type, spine_hus)
    for roi in rois:
        one_hot_roi_label = roi[:, coronal_vals, range(len(coronal_vals))]
        one_hot_roi_label = zoom(one_hot_roi_label, (1, zoom_factor), order=1).round()
        one_hot_cor_label = np.concatenate(
            (
                one_hot_cor_label,
                one_hot_roi_label.reshape(
                    (one_hot_roi_label.shape[0], one_hot_roi_label.shape[1], 1)
                ),
            ),
            axis=2,
        )

    # flip both axes of sagittal image
    sagittal_image = np.flip(sagittal_image, axis=0)    # 上下翻转
    sagittal_image = np.flip(sagittal_image, axis=1)

    # flip both axes of sagittal label
    one_hot_sag_label = np.flip(one_hot_sag_label, axis=0)
    one_hot_sag_label = np.flip(one_hot_sag_label, axis=1)

    coronal_image = np.transpose(coronal_image)
    one_hot_cor_label = np.transpose(one_hot_cor_label, (1, 0, 2))

    # flip both axes of coronal image
    coronal_image = np.flip(coronal_image, axis=0)
    coronal_image = np.flip(coronal_image, axis=1)

    # flip both axes of coronal label
    one_hot_cor_label = np.flip(one_hot_cor_label, axis=0)
    one_hot_cor_label = np.flip(one_hot_cor_label, axis=1)

    if format == "png":
        sagittal_center_name = "spine_sagittal_center.png"
        coronal_center_name = "spine_coronal_center.png"
        sagittal_updown_name = "spine_sagittal_updown.png"
        coronal_updown_name = "spine_coronal_updown.png"
    elif format == "dcm":
        sagittal_center_name = "spine_sagittal_center.dcm"
        coronal_center_name = "spine_coronal_center.dcm"
        sagittal_updown_name = "spine_sagittal_updown.dcm"
        coronal_updown_name = "spine_coronal_updown.dcm"
    else:
        raise ValueError("Format must be either png or dcm")

    img_sagittal = spine_visualization.spine_binary_segmentation_overlay(
        sagittal_image,
        one_hot_sag_label,
        output_dir,
        sagittal_center_name,
        centroids_3d,
        spine_hus=spine_hus,
        model_type=model_type,
        pixel_spacing=pixel_spacing,
    )
    img_coronal = spine_visualization.spine_binary_segmentation_overlay(
        coronal_image,
        one_hot_cor_label,
        output_dir,
        coronal_center_name,
        centroids_3d,
        spine_hus=spine_hus,
        model_type=model_type,
        pixel_spacing=pixel_spacing,
    )

    img_sagittal_updown = spine_visualization.spine_binary_segmentation_line(
        sagittal_image,
        one_hot_sag_label,
        output_dir,
        sagittal_updown_name,
        centroids_3d,
        updown_positions,
        spine_hus=spine_hus,
        model_type=model_type,
        pixel_spacing=pixel_spacing,
    )

    return img_sagittal, img_coronal, img_sagittal_updown


def curved_planar_reformation(mvs, centroids):
    centroids = sorted(centroids, key=lambda x: x[2])   # 基于层进行排序
    centroids = [(int(x[0]), int(x[1]), int(x[2])) for x in centroids]     # 全部转化为整型
    sagittal_centroids = [centroids[i][0] for i in range(0, len(centroids))]   # 矢状位上的对应层
    coronal_centroids = [centroids[i][1] for i in range(0, len(centroids))]
    axial_centroids = [centroids[i][2] for i in range(0, len(centroids))]
    sagittal_vals = [sagittal_centroids[0]] * axial_centroids[0]   # 前面的层，没有脊骨相关标签，先保留
    coronal_vals = [coronal_centroids[0]] * axial_centroids[0]

    for i in range(1, len(axial_centroids)):
        num = axial_centroids[i] - axial_centroids[i - 1]    # 两个脊柱之间的差值的层
        interp = list(np.linspace(sagittal_centroids[i - 1], sagittal_centroids[i], num=num))
        sagittal_vals.extend(interp)
        interp = list(np.linspace(coronal_centroids[i - 1], coronal_centroids[i], num=num))
        coronal_vals.extend(interp)

    sagittal_vals.extend([sagittal_centroids[-1]] * (mvs.shape[2] - len(sagittal_vals)))
    coronal_vals.extend([coronal_centroids[-1]] * (mvs.shape[2] - len(coronal_vals)))
    sagittal_vals = np.array(sagittal_vals)
    coronal_vals = np.array(coronal_vals)
    sagittal_vals = sagittal_vals.astype(int)
    coronal_vals = coronal_vals.astype(int)

    return (sagittal_vals, coronal_vals)