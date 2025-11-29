"""
基于连通区域分析的字符分割模块 (V3)
使用 Canny 边缘检测 + 连通区域分析进行字符分割
改进自 char_spilt.py 的算法

主要特点：
1. 使用 skimage.feature.canny 边缘检测（与原始 char_spilt.py 一致）
2. 连通区域分析提取候选字符区域
3. 智能合并被分割的字符（如汉字上下部分）
4. 智能拆分粘连的字符
5. 使用上半部分投影避免铆钉干扰
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# 尝试导入 skimage，如果失败则使用 OpenCV 替代
try:
    from skimage.feature import canny as skimage_canny
    from skimage.morphology import dilation as skimage_dilation
    from skimage.color import rgb2gray as skimage_rgb2gray
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def preprocess_for_canny(plate_img: np.ndarray) -> np.ndarray:
    """
    预处理车牌图像用于 Canny 边缘检测
    
    Args:
        plate_img: 输入的车牌图像 (BGR 或灰度)
        
    Returns:
        灰度图像（归一化到 0-1，与 skimage 一致）
    """
    # 添加轻微高斯模糊，减少噪点，改善边缘检测效果
    if len(plate_img.shape) == 3:
        plate_img = cv2.GaussianBlur(plate_img, (3, 3), 0)
    
    if len(plate_img.shape) == 3:
        if plate_img.shape[2] == 4:
            # RGBA -> RGB
            plate_img = plate_img[:, :, :3]
        
        # 将 BGR 转为 RGB，然后用 BGR2GRAY 处理
        # 这相当于交换 R 和 B 通道的权重：0.299B + 0.587G + 0.114R
        # 对蓝色车牌效果更好（蓝色背景与白色字符对比度更高）
        # 经测试：这种方式能多检测到 1 个字符
        rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    else:
        if plate_img.dtype == np.uint8:
            gray = plate_img.astype(np.float64) / 255.0
        else:
            gray = plate_img.copy()
    
    return gray


def canny_edge_detection(gray_img: np.ndarray, 
                         sigma: float = 3.0) -> np.ndarray:
    """
    Canny 边缘检测（优先使用 skimage.feature.canny 与原始 char_spilt.py 一致）
    
    Args:
        gray_img: 灰度图像（归一化到 0-1）
        sigma: 高斯模糊的 sigma 值
        
    Returns:
        边缘二值图像 (uint8, 0 或 255)
    """
    if HAS_SKIMAGE:
        # 使用 skimage.feature.canny（与原始 char_spilt.py 完全一致）
        edges_bool = skimage_canny(gray_img, sigma=sigma)
        
        # 膨胀操作连接断开的边缘（与原始 char_spilt.py 一致）
        edges_dilated = skimage_dilation(edges_bool)
        
        # 转换为 uint8
        edges = (edges_dilated * 255).astype(np.uint8)
    else:
        # 如果没有 skimage，抛出错误
        raise ImportError("scikit-image is required for V3 segmentation. "
                          "Please install it with: pip install scikit-image")
    
    return edges


def find_connected_regions(binary_img: np.ndarray, 
                           img_shape: Tuple[int, int]) -> List[List[int]]:
    """
    使用 skimage 连通区域分析找到字符候选区域
    
    Args:
        binary_img: 二值图像 (bool 或 uint8)
        img_shape: (height, width) 原图尺寸
        
    Returns:
        边界框列表 [[minr, minc, maxr, maxc], ...]
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for V3 segmentation.")
    
    from skimage.measure import label, regionprops
    
    h, w = img_shape
    
    # 确保是 bool 类型用于 skimage
    if binary_img.dtype == np.uint8:
        binary_bool = binary_img > 0
    else:
        binary_bool = binary_img
    
    # 使用 skimage 进行连通区域标记
    label_img = label(binary_bool)
    regions = regionprops(label_img)
    
    bboxes = []
    for props in regions:
        y0, x0 = props.centroid
        minr, minc, maxr, maxc = props.bbox
        width = maxc - minc
        height = maxr - minr
        
        # 过滤条件（与原始 char_spilt.py 完全一致）
        # 1. 宽度不能超过图像宽度的1/5
        if width > w / 5:
            continue
        # 2. 高度至少是图像高度的1/10（放宽，让拆分的部分也能保留）
        if height < h / 10:
            continue
        # 3. 宽度/高度至少有一定像素（过滤噪点）
        if width < 3 or height < 3:
            continue
        # 4. 垂直位置不能太偏离中心
        if abs(y0 - h / 2) > h * 0.45:
            continue
        
        # 检查是否被已有的 bbox 包含（与 char_spilt.py 的 in_bboxes 一致）
        bbox = [minr, minc, maxr, maxc]
        is_contained = False
        for bb in bboxes:
            minr0, minc0, maxr0, maxc0 = bb
            if minr >= minr0 and maxr <= maxr0 and minc >= minc0 and maxc <= maxc0:
                is_contained = True
                break
        
        if not is_contained:
            bboxes.append(bbox)
    
    return bboxes


def merge_vertical_bboxes(bboxes: List[List[int]], 
                          img_height: int,
                          horizontal_overlap_thresh: float = 0.3, 
                          vertical_gap_thresh: Optional[float] = None) -> List[List[int]]:
    """
    合并水平位置接近且垂直方向相邻的边界框
    处理汉字被分割成上下两部分的情况
    
    Args:
        bboxes: 边界框列表 [[minr, minc, maxr, maxc], ...]
        img_height: 图像高度
        horizontal_overlap_thresh: 水平重叠阈值
        vertical_gap_thresh: 垂直间隙阈值
        
    Returns:
        合并后的边界框列表
    """
    if len(bboxes) == 0:
        return bboxes
    
    # 动态设置垂直间隙阈值（图像高度的30%）
    if vertical_gap_thresh is None:
        vertical_gap_thresh = img_height * 0.3
    
    changed = True
    while changed:
        changed = False
        # 按水平位置排序
        bboxes = sorted(bboxes, key=lambda x: (x[1], x[0]))
        new_bboxes = []
        used = set()
        
        for i in range(len(bboxes)):
            if i in used:
                continue
                
            minr1, minc1, maxr1, maxc1 = bboxes[i]
            
            for j in range(i + 1, len(bboxes)):
                if j in used:
                    continue
                    
                minr2, minc2, maxr2, maxc2 = bboxes[j]
                
                # 计算水平重叠或接近程度
                center1 = (minc1 + maxc1) / 2
                center2 = (minc2 + maxc2) / 2
                width1 = maxc1 - minc1
                width2 = maxc2 - minc2
                avg_width = (width1 + width2) / 2
                
                # 水平中心距离小于平均宽度的一半，认为是同一列
                horizontal_close = abs(center1 - center2) < avg_width * 0.7
                
                # 或者有水平重叠
                overlap_left = max(minc1, minc2)
                overlap_right = min(maxc1, maxc2)
                horizontal_overlap = overlap_right > overlap_left
                
                if horizontal_close or horizontal_overlap:
                    # 检查垂直间距
                    vertical_gap = max(minr1, minr2) - min(maxr1, maxr2)
                    
                    if vertical_gap < vertical_gap_thresh:
                        # 合并两个框
                        minr1 = min(minr1, minr2)
                        minc1 = min(minc1, minc2)
                        maxr1 = max(maxr1, maxr2)
                        maxc1 = max(maxc1, maxc2)
                        used.add(j)
                        changed = True
            
            new_bboxes.append([minr1, minc1, maxr1, maxc1])
            used.add(i)
        
        bboxes = new_bboxes
    
    return bboxes


def split_wide_bboxes(bboxes: List[List[int]],
                      binary_img: np.ndarray,
                      img_height: int,
                      expected_char_width: Optional[float] = None) -> List[List[int]]:
    """
    拆分过宽的边界框（处理两个靠近的字符被识别为一个的情况）
    使用垂直投影法找到分割点，使用上半部分避免铆钉干扰
    """
    if len(bboxes) == 0:
        return bboxes

    # 估计单个字符的平均宽度
    widths = [b[3] - b[1] for b in bboxes]
    heights = [b[2] - b[0] for b in bboxes]

    if expected_char_width is None:
        # 过滤掉明显过宽的框，用剩余框估计正常字符宽度
        median_height = np.median(heights)
        normal_widths = [w for w in widths if w < median_height * 1.3]
        if len(normal_widths) > 0:
            expected_char_width = np.median(normal_widths)
        else:
            expected_char_width = median_height * 0.8

    new_bboxes = []

    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        width = maxc - minc
        height = maxr - minr

        # ★ 阈值从 1.4 调低到 1.25，更积极地拆宽框
        if width > expected_char_width * 1.25:
            num_chars = round(width / expected_char_width)
            if num_chars < 2:
                new_bboxes.append(bbox)
                continue

            # 使用垂直投影法找到分割点
            roi = binary_img[minr:maxr, minc:maxc]

            # 只使用上半部分计算投影（避免铆钉干扰）
            upper_ratio = 0.6
            upper_height = int(height * upper_ratio)
            if upper_height > 0:
                upper_roi = roi[:upper_height, :]
                vertical_projection = np.sum(upper_roi, axis=0)
            else:
                vertical_projection = np.sum(roi, axis=0)

            # 找到分割点
            split_points = []
            for i in range(1, num_chars):
                expected_pos = int(width * i / num_chars)
                search_range = int(width * 0.20)
                search_start = max(0, expected_pos - search_range)
                search_end = min(width, expected_pos + search_range)

                if search_start < search_end:
                    local_proj = vertical_projection[search_start:search_end]
                    if len(local_proj) > 0:
                        min_idx = search_start + np.argmin(local_proj)
                        split_points.append(min_idx)

            # 根据分割点创建新的边界框
            if len(split_points) > 0:
                split_points = sorted(set(split_points))
                prev = 0
                for sp in split_points:
                    if sp > prev + 3:
                        new_bboxes.append([minr, minc + prev, maxr, minc + sp])
                    prev = sp
                if width > prev + 3:
                    new_bboxes.append([minr, minc + prev, maxr, maxc])
            else:
                # 如果投影法没找到好的分割点，直接等分
                char_width = width // num_chars
                for i in range(num_chars):
                    start_col = minc + i * char_width
                    end_col = minc + (i + 1) * char_width if i < num_chars - 1 else maxc
                    new_bboxes.append([minr, start_col, maxr, end_col])
        else:
            new_bboxes.append(bbox)

    return new_bboxes



def filter_by_height(bboxes: List[List[int]], 
                     img_height: int, 
                     min_height_ratio: float = 0.25) -> List[List[int]]:
    """
    根据高度过滤边界框
    
    Args:
        bboxes: 边界框列表
        img_height: 图像高度
        min_height_ratio: 最小高度占比
        
    Returns:
        过滤后的边界框列表
    """
    min_height = img_height * min_height_ratio
    return [b for b in bboxes if (b[2] - b[0]) >= min_height]


def segment_characters_v3(plate_img: np.ndarray,
                          expected_chars: int = 7,
                          debug: bool = False) -> Tuple[List[np.ndarray], Optional[dict]]:
    """
    基于连通区域分析的字符分割（V3版本）
    """
    if plate_img is None or plate_img.size == 0:
        return [], None

    h, w = plate_img.shape[:2]

    # 1. 预处理
    gray = preprocess_for_canny(plate_img)

    # 2. Canny 边缘检测 + 膨胀
    edges = canny_edge_detection(gray, sigma=3.0)

    # 3. 连通区域分析找到候选区域
    bboxes = find_connected_regions(edges, (h, w))

    if len(bboxes) == 0:
        # 如果连通区域分析失败，尝试更小的 sigma
        edges = canny_edge_detection(gray, sigma=2.0)
        bboxes = find_connected_regions(edges, (h, w))

    if len(bboxes) == 0:
        if debug:
            return [], {'edges': edges, 'gray': gray, 'bboxes': []}
        return [], None

    # 4. 合并垂直方向相邻的边界框（处理汉字被拆分的问题）
    bboxes = merge_vertical_bboxes(bboxes, h)

    # 5. 拆分过宽的边界框（处理字符粘连的问题）
    bboxes = split_wide_bboxes(bboxes, edges, h)

    # 6. 高度过滤
    bboxes = filter_by_height(bboxes, h, min_height_ratio=0.25)

    # 7. 过滤边框白条
    filtered_bboxes = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        box_w = maxc - minc
        box_h = maxr - minr

        is_right_border = (
            maxc > w * 0.95 and
            box_w < w * 0.04 and
            box_h > h * 0.5
        )
        is_left_border = (
            minc < w * 0.05 and
            box_w < w * 0.04 and
            box_h > h * 0.5
        )

        if is_right_border or is_left_border:
            continue

        filtered_bboxes.append(bbox)
    bboxes = filtered_bboxes

    # 8. 按水平位置排序
    bboxes = sorted(bboxes, key=lambda x: x[1])

    # ★ 8.1 分割数量明显太少时，对最宽的框再尝试拆分一次
    if len(bboxes) < expected_chars:
        widths = [b[3] - b[1] for b in bboxes]
        heights = [b[2] - b[0] for b in bboxes]
        median_height = np.median(heights)
        normal_widths = [w0 for w0 in widths if w0 < median_height * 1.3]
        if len(normal_widths) > 0:
            expected_char_width = np.median(normal_widths)
        else:
            expected_char_width = median_height * 0.8

        # 按宽度从大到小尝试拆分，直到数量接近 expected_chars
        idx_sorted = np.argsort(widths)[::-1]
        new_bboxes = bboxes[:]
        for idx in idx_sorted:
            if len(new_bboxes) >= expected_chars:
                break
            bbox = bboxes[idx]
            sub_boxes = split_wide_bboxes([bbox], edges, h,
                                          expected_char_width=expected_char_width)
            if len(sub_boxes) > 1:
                # 用拆分结果替换原 bbox
                tmp = []
                for b in new_bboxes:
                    if b is bbox:
                        tmp.extend(sub_boxes)
                    else:
                        tmp.append(b)
                new_bboxes = tmp
        bboxes = sorted(new_bboxes, key=lambda x: x[1])

    # 8.2 如果数量太多，只取高度最大的 expected_chars 个
    if len(bboxes) > expected_chars + 2:
        heights = [(b[2] - b[0], i) for i, b in enumerate(bboxes)]
        heights.sort(reverse=True)
        keep_indices = sorted([h0[1] for h0 in heights[:expected_chars]])
        bboxes = [bboxes[i] for i in keep_indices]

    # 9. 提取字符图像
    chars = []

    if gray.dtype != np.uint8:
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = gray

    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        box_w = maxc - minc
        box_h = maxr - minr

        # 对于太窄的字符（如 "1"），添加水平 padding
        aspect_ratio = box_w / (box_h + 1e-6)
        if aspect_ratio < 0.35:
            target_w = int(box_h * 0.4)
            pad_w = (target_w - box_w) // 2
            minc = max(0, minc - pad_w)
            maxc = min(w, maxc + pad_w)

        char_img = gray_uint8[minr:maxr, minc:maxc]
        if char_img.size > 0:
            chars.append(char_img)

    debug_info = None
    if debug:
        debug_info = {
            'edges': edges,
            'gray': gray,
            'bboxes': bboxes
        }

    return chars, debug_info



def segment_with_fallback(plate_img: np.ndarray,
                          expected_chars: int = 7) -> List[np.ndarray]:
    """
    带回退机制的分割：先尝试 V3，如果不可靠则使用回退方案
    """
    # 先用 V3
    chars, _ = segment_characters_v3(plate_img, expected_chars)

    # ★ 只有当数量在一个合理范围内时才认为 V3 成功
    #    普通车牌一般期望 7 个，允许 ±2 的波动
    if expected_chars - 1 <= len(chars) <= expected_chars + 2:
        return chars

    # V3 结果数量明显不对，使用等宽分割回退
    return fixed_width_fallback(plate_img, expected_chars)



def fixed_width_fallback(plate_img: np.ndarray, 
                         expected_chars: int = 7) -> List[np.ndarray]:
    """
    等宽分割回退方案
    
    Args:
        plate_img: 车牌图像
        expected_chars: 预期字符数量
        
    Returns:
        字符图像列表
    """
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()
    
    h, w = gray.shape
    
    # 去除上下边框
    top = int(h * 0.08)
    bottom = int(h * 0.92)
    
    # 等宽分割
    char_width = int(w * 0.9 / expected_chars)
    start_x = int(w * 0.05)
    
    chars = []
    for i in range(expected_chars):
        s = start_x + i * char_width
        e = s + char_width
        s = max(0, min(s, w - 1))
        e = max(s + 1, min(e, w))
        
        char_img = gray[top:bottom, s:e]
        chars.append(char_img)
    
    return chars


def visualize_segmentation_v3(plate_img: np.ndarray, 
                              chars: List[np.ndarray],
                              debug_info: Optional[dict] = None) -> None:
    """
    可视化 V3 分割结果
    
    Args:
        plate_img: 原车牌图像
        chars: 字符图像列表
        debug_info: 调试信息
    """
    import matplotlib.pyplot as plt
    
    num_plots = 2 + len(chars)
    if debug_info and 'edges' in debug_info:
        num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=(3 * num_plots, 3))
    
    idx = 0
    
    # 原图
    if len(plate_img.shape) == 3:
        axes[idx].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
    else:
        axes[idx].imshow(plate_img, cmap='gray')
    axes[idx].set_title('原车牌')
    axes[idx].axis('off')
    idx += 1
    
    # 边缘检测结果
    if debug_info and 'edges' in debug_info:
        axes[idx].imshow(debug_info['edges'], cmap='gray')
        axes[idx].set_title('Canny边缘')
        
        # 绘制边界框
        if 'bboxes' in debug_info:
            for bbox in debug_info['bboxes']:
                minr, minc, maxr, maxc = bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
                axes[idx].add_patch(rect)
        axes[idx].axis('off')
        idx += 1
    
    # 灰度图带边界框
    if debug_info and 'gray' in debug_info:
        axes[idx].imshow(debug_info['gray'], cmap='gray')
        axes[idx].set_title('分割结果')
        
        if 'bboxes' in debug_info:
            for i, bbox in enumerate(debug_info['bboxes']):
                minr, minc, maxr, maxc = bbox
                rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='lime', linewidth=2)
                axes[idx].add_patch(rect)
                axes[idx].text(minc, minr - 2, str(i + 1), color='lime', fontsize=10)
        axes[idx].axis('off')
        idx += 1
    
    # 各字符
    for i, char_img in enumerate(chars):
        axes[idx].imshow(char_img, cmap='gray')
        axes[idx].set_title(f'字符{i + 1}')
        axes[idx].axis('off')
        idx += 1
    
    plt.tight_layout()
    plt.show()


# 导出的主要函数
__all__ = [
    'segment_characters_v3',
    'segment_with_fallback',
    'visualize_segmentation_v3',
]


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            chars, debug_info = segment_characters_v3(img, debug=True)
            print(f"分割出 {len(chars)} 个字符")
            visualize_segmentation_v3(img, chars, debug_info)
        else:
            print(f"无法读取图像: {sys.argv[1]}")
    else:
        print("用法: python segment_v3.py <车牌图像路径>")
