"""
字符分割模块
使用垂直投影法将车牌图像分割为单个字符
"""

import cv2
import numpy as np
from typing import List, Tuple


def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    预处理车牌图像：灰度化 + 二值化
    """
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    # 自适应二值化（适应不同光照）
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def vertical_projection(binary_img: np.ndarray) -> np.ndarray:
    """
    计算垂直投影直方图
    """
    return np.sum(binary_img, axis=0)


def find_split_points(projection: np.ndarray, min_gap: int = 2) -> List[Tuple[int, int]]:
    """
    根据垂直投影找到字符分割点
    返回每个字符的 (start, end) 列索引
    """
    threshold = np.max(projection) * 0.1  # 投影阈值
    
    in_char = False
    start = 0
    segments = []
    
    for i, val in enumerate(projection):
        if val > threshold and not in_char:
            in_char = True
            start = i
        elif val <= threshold and in_char:
            in_char = False
            if i - start >= min_gap:
                segments.append((start, i))
    
    # 处理最后一个字符
    if in_char:
        segments.append((start, len(projection)))
    
    return segments


def merge_segments(segments: List[Tuple[int, int]], img_width: int, 
                   expected_chars: int = 7) -> List[Tuple[int, int]]:
    """
    合并过于细碎的分割（如汉字被拆分）
    并根据预期字符数调整
    """
    if len(segments) <= expected_chars:
        return segments
    
    # 计算平均字符宽度
    avg_width = img_width / expected_chars
    
    merged = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        width = end - start
        
        # 如果当前段太窄，尝试与下一段合并
        while width < avg_width * 0.5 and i + 1 < len(segments):
            i += 1
            _, end = segments[i]
            width = end - start
        
        merged.append((start, end))
        i += 1
    
    return merged


def segment_by_equal_width(plate_img: np.ndarray, num_chars: int = 7) -> List[np.ndarray]:
    """
    等宽分割法（备用方案）
    适用于投影法失败的情况
    """
    h, w = plate_img.shape[:2]
    char_width = w // num_chars
    
    chars = []
    for i in range(num_chars):
        start = i * char_width
        end = start + char_width if i < num_chars - 1 else w
        char_img = plate_img[:, start:end]
        chars.append(char_img)
    
    return chars


def segment_characters(plate_img: np.ndarray, 
                       expected_chars: int = 7,
                       use_projection: bool = True) -> List[np.ndarray]:
    """
    主分割函数：将车牌图像分割为单个字符图像
    
    Args:
        plate_img: 矫正后的车牌图像 (BGR or Gray)
        expected_chars: 预期字符数量（普通车牌7个，新能源8个）
        use_projection: 是否使用投影法（否则用等宽分割）
    
    Returns:
        字符图像列表（灰度图）
    """
    if plate_img is None or plate_img.size == 0:
        return []
    
    # 预处理
    binary = preprocess_plate(plate_img)
    h, w = binary.shape
    
    # 裁剪上下边框（车牌边框通常占5-10%）
    top_crop = int(h * 0.1)
    bottom_crop = int(h * 0.9)
    binary_cropped = binary[top_crop:bottom_crop, :]
    
    if not use_projection:
        return segment_by_equal_width(plate_img, expected_chars)
    
    # 垂直投影分割
    projection = vertical_projection(binary_cropped)
    segments = find_split_points(projection, min_gap=3)
    
    # 如果分割结果不合理，使用等宽分割
    if len(segments) < expected_chars - 1 or len(segments) > expected_chars + 3:
        return segment_by_equal_width(plate_img, expected_chars)
    
    # 合并过于细碎的分割
    segments = merge_segments(segments, w, expected_chars)
    
    # 提取字符图像
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    
    chars = []
    for start, end in segments:
        # 稍微扩展边界
        start = max(0, start - 2)
        end = min(w, end + 2)
        char_img = gray[:, start:end]
        chars.append(char_img)
    
    return chars


def visualize_segmentation(plate_img: np.ndarray, 
                           chars: List[np.ndarray]) -> np.ndarray:
    """
    可视化分割结果
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, len(chars) + 1, figsize=(15, 3))
    
    axes[0].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原车牌')
    axes[0].axis('off')
    
    for i, char_img in enumerate(chars):
        axes[i + 1].imshow(char_img, cmap='gray')
        axes[i + 1].set_title(f'字符{i + 1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 测试代码
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        chars = segment_characters(img)
        print(f"分割出 {len(chars)} 个字符")
        visualize_segmentation(img, chars)
        import matplotlib.pyplot as plt
        plt.show()
