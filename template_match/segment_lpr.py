"""
基于 License-Plate-Recognition 项目的字符分割算法
采用水平投影 + 垂直投影 + 最大中心距分割

该算法的步骤：
1. 灰度化 + 二值化
2. 判断字符颜色，确保白字黑底
3. 形态学膨胀连接断裂区域
4. 汉字区域额外膨胀
5. 水平投影确定上下边界
6. 垂直投影 + 最大中心距分割字符
"""

import cv2
import numpy as np
from typing import List, Tuple


def load_image(image: np.ndarray) -> np.ndarray:
    """灰度化处理"""
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    return gray_image


def binarize_image(image: np.ndarray) -> np.ndarray:
    """二值化处理（Otsu）"""
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def detect_character_color(binary_image: np.ndarray) -> str:
    """
    判断二值化图像中字符的颜色
    返回 'white' 表示字符为白色（背景黑）
    返回 'black' 表示字符为黑色（背景白）
    """
    white_pixel_count = cv2.countNonZero(binary_image)
    black_pixel_count = binary_image.size - white_pixel_count
    
    if white_pixel_count > black_pixel_count:
        return 'white'
    else:
        return 'black'


def connect_characters(binary_image: np.ndarray) -> np.ndarray:
    """使用形态学膨胀操作连接字符内部的断裂区域"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected_image = cv2.dilate(binary_image, kernel, iterations=1)
    return connected_image


def connect_characters_double(image: np.ndarray) -> np.ndarray:
    """
    对前 1/8 区域（汉字区域）进行额外膨胀
    用于连接汉字的断裂区域
    """
    height, width = image.shape
    end_col = width // 8
    roi = image[0:height, 0:end_col]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated_roi = cv2.dilate(roi, kernel, iterations=1)
    
    result_image = image.copy()
    result_image[0:height, 0:end_col] = dilated_roi
    
    return result_image


def find_split_positions_horizontal(projection: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """找到水平方向的字符边界（上下）"""
    split_positions = []
    start = None
    
    for i in range(len(projection)):
        if projection[i] > threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                split_positions.append((start, i - 1))
                start = None
    
    if start is not None:
        split_positions.append((start, len(projection) - 1))
    
    return split_positions


def horizontal_projection(binary_image: np.ndarray, bi_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    水平投影，判断字符的具体范围（上下边界）
    binary_image: 膨胀后的二值化图像
    bi_image: 未膨胀的二值化图像
    """
    horizontal_sum = np.sum(binary_image, axis=1)
    horizontal_threshold = max(horizontal_sum) // 6
    horizontal_split_positions = find_split_positions_horizontal(horizontal_sum, horizontal_threshold)
    
    if not horizontal_split_positions:
        return binary_image, bi_image
    
    start, end = max(horizontal_split_positions, key=lambda x: x[1] - x[0])
    
    horizontal_area = binary_image[start:end+1, :]
    image_for_split = bi_image[start:end+1, :]
    
    return horizontal_area, image_for_split


def vertical_projection(binary_image: np.ndarray) -> np.ndarray:
    """垂直方向投影"""
    vertical_sum = np.sum(binary_image, axis=0)
    return vertical_sum


def find_split_positions_vertical(vertical_proj: np.ndarray) -> List[Tuple[int, int]]:
    """初步分割，找出字符区域"""
    threshold = np.max(vertical_proj) * 0.2
    in_character = False
    character_regions = []
    start = 0
    
    for i, value in enumerate(vertical_proj):
        if value > threshold and not in_character:
            in_character = True
            start = i
        elif value <= threshold and in_character:
            in_character = False
            character_regions.append((start, i - 1))
    
    if in_character:
        character_regions.append((start, len(vertical_proj) - 1))
    
    return character_regions


def calculate_max_center_distance(character_regions: List[Tuple[int, int]]) -> int:
    """计算字符间的最大中心距"""
    if len(character_regions) < 2:
        return 30  # 默认值
    
    centers = [(start + end) // 2 for start, end in character_regions]
    max_distance = 0
    
    for i in range(len(centers) - 1):
        distance = centers[i + 1] - centers[i]
        max_distance = max(max_distance, distance)
    
    return max_distance


def vertical_segmentation(bi_image: np.ndarray, image: np.ndarray, 
                          max_distance: int) -> List[np.ndarray]:
    """根据最大中心距分割字符"""
    vertical_proj = vertical_projection(image)
    
    threshold = np.max(vertical_proj) * 0.1
    in_character = False
    start = 0
    segments = []
    
    for i, value in enumerate(vertical_proj):
        if value > threshold and not in_character:
            in_character = True
            start = i
        elif value <= threshold and in_character:
            in_character = False
            # 过滤太窄的区域
            if (i - start > max_distance // 7) and np.max(vertical_proj[start:i]) > 5 * threshold:
                segments.append(bi_image[:, start:i])
    
    # 处理最后一个字符
    if in_character and len(vertical_proj) - start > max_distance // 3:
        segments.append(bi_image[:, start:])
    
    return segments


def split_char(image: np.ndarray) -> List[np.ndarray]:
    """
    主分割函数
    
    Args:
        image: 车牌图像 (BGR 或 灰度)
    
    Returns:
        分割后的字符图像列表（黑底白字）
    """
    # 1. 灰度化 + 二值化
    gray_image = load_image(image)
    bi_image = binarize_image(gray_image)
    
    # 2. 判断字体颜色，确保白字黑底
    if detect_character_color(bi_image) == 'white':
        bi_image = cv2.bitwise_not(bi_image)
    
    # 3. 膨胀连接断裂区域
    binary_image = connect_characters(bi_image)
    
    # 4. 汉字区域额外膨胀
    binary_image = connect_characters_double(binary_image)
    
    # 5. 水平投影确定上下边界
    horizontal_proj, bi_image_cropped = horizontal_projection(binary_image, bi_image)
    
    # 6. 垂直投影分割
    vertical_proj = vertical_projection(horizontal_proj)
    
    # 7. 初步分割确定最大中心距
    character_regions = find_split_positions_vertical(vertical_proj)
    max_distance = calculate_max_center_distance(character_regions)
    
    # 8. 最终分割
    segments = vertical_segmentation(bi_image_cropped, horizontal_proj, max_distance)
    
    return segments


def segment_characters_lpr(plate_img: np.ndarray, expected_chars: int = 7) -> List[np.ndarray]:
    """
    兼容接口：使用 LPR 项目的分割算法
    
    Args:
        plate_img: 车牌图像
        expected_chars: 期望的字符数量（用于验证）
    
    Returns:
        分割后的字符图像列表
    """
    segments = split_char(plate_img)
    
    # 如果分割结果不合理，返回空列表
    if len(segments) < 2:
        return []
    
    return segments


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            segments = split_char(img)
            print(f"分割出 {len(segments)} 个字符")
            
            # 显示结果
            fig, axes = plt.subplots(1, len(segments) + 1, figsize=(12, 3))
            
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            for i, seg in enumerate(segments):
                axes[i + 1].imshow(seg, cmap='gray')
                axes[i + 1].set_title(f'Char {i+1}')
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            plt.savefig('lpr_split_result.png', dpi=150)
            plt.show()
