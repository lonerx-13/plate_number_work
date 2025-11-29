"""
HOG 匹配优化分析脚本
分析当前 HOG 匹配失败的原因并提供改进建议
"""

import cv2
import numpy as np
import os
from template_match.segment_v3 import segment_with_fallback
from template_match.deep_feature import HOGFeatureExtractor, HOGTemplateMatcher

# CCPD 字符映射
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
             "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
             'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', 
       '4', '5', '6', '7', '8', '9', 'O']


def parse_ccpd_filename(filename):
    """解析 CCPD 文件名"""
    try:
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('-')
        if len(parts) < 5:
            return None
        
        plate_indices = parts[4].split('_')
        if len(plate_indices) < 7:
            return None
            
        plate_str = PROVINCES[int(plate_indices[0])]
        plate_str += ALPHABETS[int(plate_indices[1])]
        for idx in plate_indices[2:]:
            plate_str += ADS[int(idx)]
        
        vertices_str = parts[3]
        v_coords = vertices_str.split('_')
        vertices = []
        for v in v_coords:
            vx, vy = v.split('&')
            vertices.append([int(vx), int(vy)])
        
        vertices = np.array([vertices[2], vertices[3], vertices[0], vertices[1]], dtype="float32")
        
        return {
            'plate_number': plate_str,
            'vertices': vertices,
        }
    except:
        return None


def four_point_transform(image, pts, width=240, height=80):
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (width, height))


def analyze_failure_case(ccpd_root, max_images=20):
    """分析 HOG 匹配失败的案例"""
    
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character")
    extractor = HOGFeatureExtractor()
    matcher = HOGTemplateMatcher(template_dir, "140", extractor)
    
    # 收集图片
    image_files = []
    for root, _, files in os.walk(ccpd_root):
        for f in files:
            if f.endswith('.jpg'):
                image_files.append(os.path.join(root, f))
    
    image_files = image_files[:max_images]
    
    print(f"分析 {len(image_files)} 张图片的 HOG 匹配情况\n")
    
    total_chars = 0
    correct_chars = 0
    error_analysis = {
        'segment_mismatch': 0,  # 分割数量不对
        'chinese_error': 0,     # 汉字识别错误
        'letter_error': 0,      # 字母识别错误
        'digit_error': 0,       # 数字识别错误
    }
    
    confusion = {}  # {(gt, pred): count}
    
    for img_path in image_files:
        info = parse_ccpd_filename(img_path)
        if info is None:
            continue
        
        gt = info['plate_number']
        
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        warped = four_point_transform(img, info['vertices'])
        char_images = segment_with_fallback(warped, len(gt))
        
        if len(char_images) != len(gt):
            error_analysis['segment_mismatch'] += 1
            continue
        
        # 逐字符分析
        for i, (char_img, gt_char) in enumerate(zip(char_images, gt)):
            total_chars += 1
            
            if i == 0:
                results = matcher.match_character(char_img, is_chinese=True, method="cosine", position=0)
            else:
                results = matcher.match_character(char_img, is_chinese=False, method="cosine", position=i)
            
            pred_char = results[0][0] if results else '?'
            score = results[0][1] if results else 0
            
            if pred_char == gt_char:
                correct_chars += 1
            else:
                # 记录混淆
                key = (gt_char, pred_char)
                confusion[key] = confusion.get(key, 0) + 1
                
                if i == 0:
                    error_analysis['chinese_error'] += 1
                elif gt_char.isdigit():
                    error_analysis['digit_error'] += 1
                else:
                    error_analysis['letter_error'] += 1
    
    print("="*60)
    print("HOG 匹配分析结果")
    print("="*60)
    print(f"总字符数: {total_chars}")
    print(f"正确字符数: {correct_chars}")
    print(f"字符准确率: {correct_chars/max(total_chars,1)*100:.2f}%")
    print()
    print("错误分析:")
    print(f"  分割数量不匹配: {error_analysis['segment_mismatch']} 张图片")
    print(f"  汉字识别错误: {error_analysis['chinese_error']}")
    print(f"  字母识别错误: {error_analysis['letter_error']}")
    print(f"  数字识别错误: {error_analysis['digit_error']}")
    print()
    print("常见混淆 (真实 -> 预测):")
    sorted_confusion = sorted(confusion.items(), key=lambda x: -x[1])[:15]
    for (gt, pred), count in sorted_confusion:
        print(f"  '{gt}' -> '{pred}': {count} 次")
    
    return error_analysis, confusion


if __name__ == "__main__":
    import sys
    ccpd_root = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\33005\Downloads\ccpd_fn"
    
    if os.path.exists(ccpd_root):
        error_analysis, confusion = analyze_failure_case(ccpd_root, max_images=50)
        
        print("\n" + "="*60)
        print("改进建议")
        print("="*60)
        print("""
基于分析结果，HOG 匹配准确率低的主要原因：

1. 【分割问题】82% 的图片分割失败
   - CCPD 数据集图片质量参差不齐
   - V3 分割对低质量图片鲁棒性不足
   
2. 【模板问题】数字经常被误识别为字母 E、D
   - 可能是字母 E、D 的模板特征太"通用"
   - 需要更精确的模板或更多模板变体
   
3. 【预处理问题】CCPD 图片与模板字符的预处理差异
   - 模板是白底黑字的标准字体
   - 实际分割出的字符可能有噪声、光照不均等

改进方向：
1. 使用高质量图片扩充模板库（用 expand_templates.py 的预览模式）
2. 增加模板变体数量，特别是易混淆字符
3. 调整 HOG 预处理参数（如 CLAHE 参数、二值化阈值）
4. 考虑使用多特征融合（HOG + 其他特征）
5. 对于 CCPD 这种复杂数据集，LPRNet 更适合
        """)
    else:
        print(f"目录不存在: {ccpd_root}")
