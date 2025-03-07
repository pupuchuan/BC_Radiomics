"""
@author: louisblankemeier
"""

import os
from pathlib import Path
from typing import Union

import numpy as np

from visualization.detectron_visualizer import Visualizer

spine_name_list = ['S', 'L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                   'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']    # 注意：数据里有些可能没有
body_name_list = ['Muscle', 'IMAT', 'VAT', 'SAT', 'Bone']

def spine_binary_segmentation_overlay(
    img_in: np.ndarray,
    mask: np.ndarray,
    base_path: Union[str, Path],
    file_name: str,
    centroids_3d: dict,
    figure_text_key=None,
    spine_hus=None,
    spine=True,
    model_type=None,
    pixel_spacing=None,
):
    """Save binary segmentation overlay.
    Args:
        img_in (Union[str, Path]): Path to the input image.
        mask (Union[str, Path]): Path to the mask.
        base_path (Union[str, Path]): Path to the output directory.
        file_name (str): Output file name.
        centroids (list, optional): List of centroids. Defaults to None.
        figure_text_key (dict, optional): Figure text key. Defaults to None.
        spine_hus (list, optional): List of HU values. Defaults to None.
        spine (bool, optional): Spine flag. Defaults to True.
        model_type (Models): Model type. Defaults to None.
    """
    # _COLORS = (
    #     np.array(
    #         [
    #             1.000,
    #             0.000,
    #             0.000,
    #             0.000,
    #             1.000,
    #             0.000,
    #             1.000,
    #             1.000,
    #             0.000,
    #             1.000,
    #             0.500,
    #             0.000,
    #             0.000,
    #             1.000,
    #             1.000,
    #             1.000,
    #             0.000,
    #             1.000,
    #         ]
    #     )
    #     .astype(np.float32)
    #     .reshape(-1, 3)
    # )         # 5个颜色
    #
    # label_map = {"L5": 0, "L4": 1, "L3": 2, "L2": 3, "L1": 4, "T12": 5}   # 对应的5个颜色

    # _COLORS = np.array([
    #     [1.000, 0.647, 0.000],  # 橙色
    #     [0.000, 1.000, 0.000],  # 绿色
    #     [0.000, 0.000, 1.000],  # 蓝色
    #     [1.000, 0.000, 1.000],  # 紫色
    #     [1.000, 0.412, 0.706],  # 粉色
    #     [0.000, 1.000, 1.000],  # 青色
    #     [0.275, 0.510, 0.706],  # 深蓝色
    #     [0.000, 0.502, 0.000],  # 鲜绿色
    #     [0.000, 0.502, 0.502],  # 水绿色
    #     [0.412, 0.412, 0.412],  # 灰色
    #     [0.753, 0.753, 0.753],  # 亮灰色
    #     [0.000, 0.000, 0.502],  # 深蓝色
    #     [0.863, 0.078, 0.235],  # 深红色
    #     [0.933, 0.510, 0.933],  # 浅紫色
    #     [0.596, 0.984, 0.596],  # 浅绿色
    #     [1.000, 0.843, 0.000],  # 金色
    #     [0.933, 0.604, 0.804],  # 浅粉色
    #     [0.941, 0.902, 0.549],  # 浅黄色
    #     [0.667, 0.000, 1.000],  # 深紫色
    #     [0.824, 0.706, 0.549],  # 棕色
    #     [0.502, 0.000, 0.502],  # 深紫色
    #     [1.000, 0.725, 0.058],  # 橙黄色
    #     [0.933, 0.910, 0.667],  # 浅棕色
    #     [0.863, 0.863, 0.863],  # 浅灰色
    #     [0.294, 0.000, 0.510]  # 深紫色
    # ])

    _COLORS = np.array([
        [1.000, 0.000, 0.000],  # 红色
        [0.000, 1.000, 0.000],  # 绿色
        [0.000, 0.000, 1.000],  # 蓝色
        [1.000, 1.000, 0.000],  # 黄色
        [0.600, 0.000, 1.000],  # 紫色
        [0.000, 1.000, 1.000],  # 青色
        [1.000, 0.500, 0.000],  # 橙色
        [1.000, 0.400, 0.800],  # 粉色
        [0.600, 0.000, 0.000],  # 深红色
        [0.000, 0.600, 0.000],  # 深绿色
        [0.000, 0.000, 0.600],  # 深蓝色
        [0.800, 0.800, 0.000],  # 深黄色
        [0.400, 0.000, 0.600],  # 深紫色
        [0.000, 0.600, 0.600],  # 深青色
        [0.800, 0.400, 0.000],  # 深橙色
        [1.000, 0.800, 0.800],  # 浅红色
        [0.800, 1.000, 0.800],  # 浅绿色
        [0.800, 0.800, 1.000],  # 浅蓝色
        [1.000, 1.000, 0.800],  # 浅黄色
        [0.800, 0.600, 1.000],  # 浅紫色
        [0.800, 1.000, 1.000],  # 浅青色
        [1.000, 0.800, 0.400],  # 浅橙色
        [0.800, 0.200, 0.000],  # 红褐色
        [0.400, 0.600, 0.200],  # 绿褐色
        [0.200, 0.400, 0.800]  # 蓝褐色
    ])

    label_map = {}
    for i, name in enumerate(spine_name_list):
        label_map[name] = i

    _ROI_COLOR = np.array([1.000, 0.340, 0.200])

    _SPINE_TEXT_OFFSET_FROM_TOP = 10.0
    _SPINE_TEXT_OFFSET_FROM_RIGHT = 63.0
    _SPINE_TEXT_VERTICAL_SPACING = 14.0

    img_in = np.clip(img_in, -300, 1800)    # 输出的窗宽窗位
    img_in = normalize_img(img_in) * 255.0
    # images_base_path = Path(base_path) / "images"
    images_base_path = Path(base_path)    # 这里不要多一个images的文件夹
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois
    for i, _ in enumerate(levels):
        color = _ROI_COLOR
        edge_color = color
        vis.draw_binary_mask(
            mask[:, :, num_levels + i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw text and lines
    for i, level in enumerate(levels):
        vis.draw_text(
            # text=f"{level}: {round(float(spine_hus[level]))}",
            text=f"{level}: {round(centroids_3d[level][-1]) + 1}",     # 表示层数，+1表示真实的层数(从1开始)
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                _SPINE_TEXT_VERTICAL_SPACING * i + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=_COLORS[label_map[level]],
            font_size=9,
            horizontal_alignment="left",
        )

        """
        vis.draw_line(
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.25,
        )
        """

    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img

def spine_binary_segmentation_line(
    img_in: np.ndarray,
    mask: np.ndarray,
    base_path: Union[str, Path],
    file_name: str,
    centroids_3d: dict,
    updown_positions: dict,
    figure_text_key=None,
    spine_hus=None,
    spine=True,
    model_type=None,
    pixel_spacing=None,
):
    """Save binary segmentation overlay.
    Args:
        img_in (Union[str, Path]): Path to the input image.
        mask (Union[str, Path]): Path to the mask.
        base_path (Union[str, Path]): Path to the output directory.
        file_name (str): Output file name.
        centroids (list, optional): List of centroids. Defaults to None.
        figure_text_key (dict, optional): Figure text key. Defaults to None.
        spine_hus (list, optional): List of HU values. Defaults to None.
        spine (bool, optional): Spine flag. Defaults to True.
        model_type (Models): Model type. Defaults to None.
    """

    _COLORS = np.array([
        [1.000, 0.000, 0.000],  # 红色
        [0.000, 1.000, 0.000],  # 绿色
        [0.000, 0.000, 1.000],  # 蓝色
        [1.000, 1.000, 0.000],  # 黄色
        [0.600, 0.000, 1.000],  # 紫色
        [0.000, 1.000, 1.000],  # 青色
        [1.000, 0.500, 0.000],  # 橙色
        [1.000, 0.400, 0.800],  # 粉色
        [0.600, 0.000, 0.000],  # 深红色
        [0.000, 0.600, 0.000],  # 深绿色
        [0.000, 0.000, 0.600],  # 深蓝色
        [0.800, 0.800, 0.000],  # 深黄色
        [0.400, 0.000, 0.600],  # 深紫色
        [0.000, 0.600, 0.600],  # 深青色
        [0.800, 0.400, 0.000],  # 深橙色
        [1.000, 0.800, 0.800],  # 浅红色
        [0.800, 1.000, 0.800],  # 浅绿色
        [0.800, 0.800, 1.000],  # 浅蓝色
        [1.000, 1.000, 0.800],  # 浅黄色
        [0.800, 0.600, 1.000],  # 浅紫色
        [0.800, 1.000, 1.000],  # 浅青色
        [1.000, 0.800, 0.400],  # 浅橙色
        [0.800, 0.200, 0.000],  # 红褐色
        [0.400, 0.600, 0.200],  # 绿褐色
        [0.200, 0.400, 0.800]  # 蓝褐色
    ])

    label_map = {}
    for i, name in enumerate(spine_name_list):
        label_map[name] = i

    _ROI_COLOR = np.array([1.000, 0.340, 0.200])

    _SPINE_TEXT_OFFSET_FROM_TOP = 10.0
    _SPINE_TEXT_OFFSET_FROM_RIGHT = 92.0   # 63.0
    _SPINE_TEXT_VERTICAL_SPACING = 14.0

    img_in = np.clip(img_in, -300, 1800)    # 输出的窗宽窗位
    img_in = normalize_img(img_in) * 255.0
    # images_base_path = Path(base_path) / "images"
    images_base_path = Path(base_path)    # 这里不要多一个images的文件夹
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois
    # for i, _ in enumerate(levels):
    #     color = _ROI_COLOR
    #     edge_color = color
    #     vis.draw_binary_mask(
    #         mask[:, :, num_levels + i].astype(int),
    #         color=color,
    #         edge_color=edge_color,
    #         alpha=alpha_val,
    #         area_threshold=0,
    #     )

    # draw text and lines
    for i, level in enumerate(levels):
        vis.draw_text(
            # text=f"{level}: {round(float(spine_hus[level]))}",
            text=f"{level}: ({updown_positions[level][0] + 1}, {updown_positions[level][1]})",     # 表示层数，+1表示真实的层数(从1开始)
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                _SPINE_TEXT_VERTICAL_SPACING * i + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=_COLORS[label_map[level]],
            font_size=9,
            horizontal_alignment="left",
        )

        vis.draw_line(                 # 两条线，一条上边界，一条下边界
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    mask.shape[0] - updown_positions[level][0] * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    mask.shape[0] - updown_positions[level][0] * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.25,
        )

        vis.draw_line(                 # 两条线，一条上边界，一条下边界
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    mask.shape[0] - updown_positions[level][1] * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    mask.shape[0] - updown_positions[level][1] * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.1,
        )


    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img

def normalize_img(img: np.ndarray) -> np.ndarray:
    """Normalize the image.
    Args:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (img - img.min()) / (img.max() - img.min())


def spine_binary_segmentation_overlay(
    img_in: np.ndarray,
    mask: np.ndarray,
    base_path: Union[str, Path],
    file_name: str,
    centroids_3d: dict,
    figure_text_key=None,
    spine_hus=None,
    spine=True,
    model_type=None,
    pixel_spacing=None,
):
    """Save binary segmentation overlay.
    Args:
        img_in (Union[str, Path]): Path to the input image.
        mask (Union[str, Path]): Path to the mask.
        base_path (Union[str, Path]): Path to the output directory.
        file_name (str): Output file name.
        centroids (list, optional): List of centroids. Defaults to None.
        figure_text_key (dict, optional): Figure text key. Defaults to None.
        spine_hus (list, optional): List of HU values. Defaults to None.
        spine (bool, optional): Spine flag. Defaults to True.
        model_type (Models): Model type. Defaults to None.
    """
    # _COLORS = (
    #     np.array(
    #         [
    #             1.000,
    #             0.000,
    #             0.000,
    #             0.000,
    #             1.000,
    #             0.000,
    #             1.000,
    #             1.000,
    #             0.000,
    #             1.000,
    #             0.500,
    #             0.000,
    #             0.000,
    #             1.000,
    #             1.000,
    #             1.000,
    #             0.000,
    #             1.000,
    #         ]
    #     )
    #     .astype(np.float32)
    #     .reshape(-1, 3)
    # )         # 5个颜色
    #
    # label_map = {"L5": 0, "L4": 1, "L3": 2, "L2": 3, "L1": 4, "T12": 5}   # 对应的5个颜色

    # _COLORS = np.array([
    #     [1.000, 0.647, 0.000],  # 橙色
    #     [0.000, 1.000, 0.000],  # 绿色
    #     [0.000, 0.000, 1.000],  # 蓝色
    #     [1.000, 0.000, 1.000],  # 紫色
    #     [1.000, 0.412, 0.706],  # 粉色
    #     [0.000, 1.000, 1.000],  # 青色
    #     [0.275, 0.510, 0.706],  # 深蓝色
    #     [0.000, 0.502, 0.000],  # 鲜绿色
    #     [0.000, 0.502, 0.502],  # 水绿色
    #     [0.412, 0.412, 0.412],  # 灰色
    #     [0.753, 0.753, 0.753],  # 亮灰色
    #     [0.000, 0.000, 0.502],  # 深蓝色
    #     [0.863, 0.078, 0.235],  # 深红色
    #     [0.933, 0.510, 0.933],  # 浅紫色
    #     [0.596, 0.984, 0.596],  # 浅绿色
    #     [1.000, 0.843, 0.000],  # 金色
    #     [0.933, 0.604, 0.804],  # 浅粉色
    #     [0.941, 0.902, 0.549],  # 浅黄色
    #     [0.667, 0.000, 1.000],  # 深紫色
    #     [0.824, 0.706, 0.549],  # 棕色
    #     [0.502, 0.000, 0.502],  # 深紫色
    #     [1.000, 0.725, 0.058],  # 橙黄色
    #     [0.933, 0.910, 0.667],  # 浅棕色
    #     [0.863, 0.863, 0.863],  # 浅灰色
    #     [0.294, 0.000, 0.510]  # 深紫色
    # ])

    _COLORS = np.array([
        [1.000, 0.000, 0.000],  # 红色
        [0.000, 1.000, 0.000],  # 绿色
        [0.000, 0.000, 1.000],  # 蓝色
        [1.000, 1.000, 0.000],  # 黄色
        [0.600, 0.000, 1.000],  # 紫色
        [0.000, 1.000, 1.000],  # 青色
        [1.000, 0.500, 0.000],  # 橙色
        [1.000, 0.400, 0.800],  # 粉色
        [0.600, 0.000, 0.000],  # 深红色
        [0.000, 0.600, 0.000],  # 深绿色
        [0.000, 0.000, 0.600],  # 深蓝色
        [0.800, 0.800, 0.000],  # 深黄色
        [0.400, 0.000, 0.600],  # 深紫色
        [0.000, 0.600, 0.600],  # 深青色
        [0.800, 0.400, 0.000],  # 深橙色
        [1.000, 0.800, 0.800],  # 浅红色
        [0.800, 1.000, 0.800],  # 浅绿色
        [0.800, 0.800, 1.000],  # 浅蓝色
        [1.000, 1.000, 0.800],  # 浅黄色
        [0.800, 0.600, 1.000],  # 浅紫色
        [0.800, 1.000, 1.000],  # 浅青色
        [1.000, 0.800, 0.400],  # 浅橙色
        [0.800, 0.200, 0.000],  # 红褐色
        [0.400, 0.600, 0.200],  # 绿褐色
        [0.200, 0.400, 0.800]  # 蓝褐色
    ])

    label_map = {}
    for i, name in enumerate(spine_name_list):
        label_map[name] = i

    _ROI_COLOR = np.array([1.000, 0.340, 0.200])

    _SPINE_TEXT_OFFSET_FROM_TOP = 10.0
    _SPINE_TEXT_OFFSET_FROM_RIGHT = 63.0
    _SPINE_TEXT_VERTICAL_SPACING = 14.0

    img_in = np.clip(img_in, -300, 1800)    # 输出的窗宽窗位
    img_in = normalize_img(img_in) * 255.0
    # images_base_path = Path(base_path) / "images"
    images_base_path = Path(base_path)    # 这里不要多一个images的文件夹
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois
    for i, _ in enumerate(levels):
        color = _ROI_COLOR
        edge_color = color
        vis.draw_binary_mask(
            mask[:, :, num_levels + i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw text and lines
    for i, level in enumerate(levels):
        vis.draw_text(
            # text=f"{level}: {round(float(spine_hus[level]))}",
            text=f"{level}: {round(centroids_3d[level][-1]) + 1}",     # 表示层数，+1表示真实的层数(从1开始)
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                _SPINE_TEXT_VERTICAL_SPACING * i + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=_COLORS[label_map[level]],
            font_size=9,
            horizontal_alignment="left",
        )

        """
        vis.draw_line(
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    inferior_superior_centers[num_levels - i - 1]
                    * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.25,
        )
        """

    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img

def spine_binary_segmentation_line(
    img_in: np.ndarray,
    mask: np.ndarray,
    base_path: Union[str, Path],
    file_name: str,
    centroids_3d: dict,
    updown_positions: dict,
    figure_text_key=None,
    spine_hus=None,
    spine=True,
    model_type=None,
    pixel_spacing=None,
):
    """Save binary segmentation overlay.
    Args:
        img_in (Union[str, Path]): Path to the input image.
        mask (Union[str, Path]): Path to the mask.
        base_path (Union[str, Path]): Path to the output directory.
        file_name (str): Output file name.
        centroids (list, optional): List of centroids. Defaults to None.
        figure_text_key (dict, optional): Figure text key. Defaults to None.
        spine_hus (list, optional): List of HU values. Defaults to None.
        spine (bool, optional): Spine flag. Defaults to True.
        model_type (Models): Model type. Defaults to None.
    """

    _COLORS = np.array([
        [1.000, 0.000, 0.000],  # 红色
        [0.000, 1.000, 0.000],  # 绿色
        [0.000, 0.000, 1.000],  # 蓝色
        [1.000, 1.000, 0.000],  # 黄色
        [0.600, 0.000, 1.000],  # 紫色
        [0.000, 1.000, 1.000],  # 青色
        [1.000, 0.500, 0.000],  # 橙色
        [1.000, 0.400, 0.800],  # 粉色
        [0.600, 0.000, 0.000],  # 深红色
        [0.000, 0.600, 0.000],  # 深绿色
        [0.000, 0.000, 0.600],  # 深蓝色
        [0.800, 0.800, 0.000],  # 深黄色
        [0.400, 0.000, 0.600],  # 深紫色
        [0.000, 0.600, 0.600],  # 深青色
        [0.800, 0.400, 0.000],  # 深橙色
        [1.000, 0.800, 0.800],  # 浅红色
        [0.800, 1.000, 0.800],  # 浅绿色
        [0.800, 0.800, 1.000],  # 浅蓝色
        [1.000, 1.000, 0.800],  # 浅黄色
        [0.800, 0.600, 1.000],  # 浅紫色
        [0.800, 1.000, 1.000],  # 浅青色
        [1.000, 0.800, 0.400],  # 浅橙色
        [0.800, 0.200, 0.000],  # 红褐色
        [0.400, 0.600, 0.200],  # 绿褐色
        [0.200, 0.400, 0.800]  # 蓝褐色
    ])

    label_map = {}
    for i, name in enumerate(spine_name_list):
        label_map[name] = i

    _ROI_COLOR = np.array([1.000, 0.340, 0.200])

    _SPINE_TEXT_OFFSET_FROM_TOP = 10.0
    _SPINE_TEXT_OFFSET_FROM_RIGHT = 92.0   # 63.0
    _SPINE_TEXT_VERTICAL_SPACING = 14.0

    img_in = np.clip(img_in, -300, 1800)    # 输出的窗宽窗位
    img_in = normalize_img(img_in) * 255.0
    # images_base_path = Path(base_path) / "images"
    images_base_path = Path(base_path)    # 这里不要多一个images的文件夹
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    # draw rois
    # for i, _ in enumerate(levels):
    #     color = _ROI_COLOR
    #     edge_color = color
    #     vis.draw_binary_mask(
    #         mask[:, :, num_levels + i].astype(int),
    #         color=color,
    #         edge_color=edge_color,
    #         alpha=alpha_val,
    #         area_threshold=0,
    #     )

    # draw text and lines
    for i, level in enumerate(levels):
        vis.draw_text(
            # text=f"{level}: {round(float(spine_hus[level]))}",
            text=f"{level}: ({updown_positions[level][0] + 1}, {updown_positions[level][1]})",     # 表示层数，+1表示真实的层数(从1开始)
            position=(
                mask.shape[1] - _SPINE_TEXT_OFFSET_FROM_RIGHT,
                _SPINE_TEXT_VERTICAL_SPACING * i + _SPINE_TEXT_OFFSET_FROM_TOP,
            ),
            color=_COLORS[label_map[level]],
            font_size=9,
            horizontal_alignment="left",
        )

        vis.draw_line(                 # 两条线，一条上边界，一条下边界
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    mask.shape[0] - updown_positions[level][0] * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    mask.shape[0] - updown_positions[level][0] * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.25,
        )

        vis.draw_line(                 # 两条线，一条上边界，一条下边界
            x_data=(0, mask.shape[1] - 1),
            y_data=(
                int(
                    mask.shape[0] - updown_positions[level][1] * (pixel_spacing[2] / pixel_spacing[1])
                ),
                int(
                    mask.shape[0] - updown_positions[level][1] * (pixel_spacing[2] / pixel_spacing[1])
                ),
            ),
            color=_COLORS[label_map[level]],
            linestyle="dashed",
            linewidth=0.1,
        )


    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img


def img_mask(
    img_in: np.ndarray,
    mask: np.ndarray,
    base_path: Union[str, Path],
    file_name: str,
    spine_hus=None,
):

    _COLORS = np.array([
        [1.000, 0.000, 0.000],  # 红色
        [0.000, 1.000, 0.000],  # 绿色
        [0.000, 0.000, 1.000],  # 蓝色
        [1.000, 1.000, 0.000],  # 黄色
        [0.600, 0.000, 1.000],  # 紫色
        [0.000, 1.000, 1.000],  # 青色
        [1.000, 0.500, 0.000],  # 橙色
        [1.000, 0.400, 0.800],  # 粉色
        [0.600, 0.000, 0.000],  # 深红色
        [0.000, 0.600, 0.000],  # 深绿色
        [0.000, 0.000, 0.600],  # 深蓝色
        [0.800, 0.800, 0.000],  # 深黄色
        [0.400, 0.000, 0.600],  # 深紫色
        [0.000, 0.600, 0.600],  # 深青色
        [0.800, 0.400, 0.000],  # 深橙色
        [1.000, 0.800, 0.800],  # 浅红色
        [0.800, 1.000, 0.800],  # 浅绿色
        [0.800, 0.800, 1.000],  # 浅蓝色
        [1.000, 1.000, 0.800],  # 浅黄色
        [0.800, 0.600, 1.000],  # 浅紫色
        [0.800, 1.000, 1.000],  # 浅青色
        [1.000, 0.800, 0.400],  # 浅橙色
        [0.800, 0.200, 0.000],  # 红褐色
        [0.400, 0.600, 0.200],  # 绿褐色
        [0.200, 0.400, 0.800]  # 蓝褐色
    ])

    label_map = {}
    for i, name in enumerate(spine_name_list):
        label_map[name] = i

    img_in = np.clip(img_in, -300, 1800)    # 输出的窗宽窗位
    img_in = normalize_img(img_in) * 255.0
    # images_base_path = Path(base_path) / "images"
    images_base_path = Path(base_path)    # 这里不要多一个images的文件夹
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img

def img_mask_body(
    img_in: np.ndarray,
    mask: np.ndarray,
    base_path: Union[str, Path],
    file_name: str,
    spine_hus=None,
):

    _COLORS = np.array([
        [1.000, 0.000, 0.000],  # 红色
        [0.000, 1.000, 0.000],  # 绿色
        [0.000, 0.000, 1.000],  # 蓝色
        [1.000, 1.000, 0.000],  # 黄色
        [0.000, 1.000, 1.000],  # 青色
        [0.600, 0.000, 1.000],  # 紫色
        [1.000, 0.500, 0.000],  # 橙色
        [1.000, 0.400, 0.800],  # 粉色
        [0.600, 0.000, 0.000],  # 深红色
        [0.000, 0.600, 0.000],  # 深绿色
        [0.000, 0.000, 0.600],  # 深蓝色
        [0.800, 0.800, 0.000],  # 深黄色
        [0.400, 0.000, 0.600],  # 深紫色
        [0.000, 0.600, 0.600],  # 深青色
        [0.800, 0.400, 0.000],  # 深橙色
        [1.000, 0.800, 0.800],  # 浅红色
        [0.800, 1.000, 0.800],  # 浅绿色
        [0.800, 0.800, 1.000],  # 浅蓝色
        [1.000, 1.000, 0.800],  # 浅黄色
        [0.800, 0.600, 1.000],  # 浅紫色
        [0.800, 1.000, 1.000],  # 浅青色
        [1.000, 0.800, 0.400],  # 浅橙色
        [0.800, 0.200, 0.000],  # 红褐色
        [0.400, 0.600, 0.200],  # 绿褐色
        [0.200, 0.400, 0.800]  # 蓝褐色
    ])

    label_map = {}
    for i, name in enumerate(body_name_list):
        label_map[name] = i

    img_in = np.clip(img_in, -125, 225)    # 输出的窗宽窗位
    img_in = normalize_img(img_in) * 255.0
    # images_base_path = Path(base_path) / "images"
    images_base_path = Path(base_path)    # 这里不要多一个images的文件夹
    images_base_path.mkdir(exist_ok=True)

    img_in = img_in.reshape((img_in.shape[0], img_in.shape[1], 1))
    img_rgb = np.tile(img_in, (1, 1, 3))

    vis = Visualizer(img_rgb)

    levels = list(spine_hus.keys())
    levels.reverse()
    num_levels = len(levels)

    # draw seg masks
    for i, level in enumerate(levels):
        color = _COLORS[label_map[level]]
        edge_color = None
        alpha_val = 0.2
        vis.draw_binary_mask(
            mask[:, :, i].astype(int),
            color=color,
            edge_color=edge_color,
            alpha=alpha_val,
            area_threshold=0,
        )

    vis_obj = vis.get_output()
    img = vis_obj.save(os.path.join(images_base_path, file_name))
    return img