"""
完整的模板匹配识别器
整合字符分割和模板匹配
支持像素匹配和 HOG 特征匹配
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from enum import Enum

from .segment_v3 import segment_with_fallback
from .matcher import TemplateMatcher, MatchMethod


class SegmentMethod(Enum):
    """分割方法枚举"""
    V3_FALLBACK = "v3_fallback"           # V3 连通区域分析（带回退机制）- 推荐


class TemplateRecognizer:
    """
    基于模板匹配的车牌识别器
    支持两种匹配方式：
    1. 像素匹配（快速，效果一般）
    2. HOG 特征匹配（对简单字符图像区分度更好）
    
    支持多种分割方法：
    - V2: 垂直投影法、混合策略
    - V3: 连通区域分析法（Canny边缘检测）
    """
    
    def __init__(self, template_dir: str = None, use_deep_feature: bool = False,
                 segment_method: SegmentMethod = SegmentMethod.V3_FALLBACK):
        """
        初始化识别器
        
        Args:
            template_dir: 模板目录路径，默认为 'character'
            use_deep_feature: 是否使用 HOG 特征匹配
            segment_method: 分割方法，默认使用 V3 带回退机制
        """
        if template_dir is None:
            # 默认在项目根目录下的 character 文件夹
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_dir = os.path.join(base_dir, "character")
        
        self.template_dir = template_dir
        self.matchers = {}  # 缓存不同类型的匹配器
        self.deep_matchers = {}  # 深度特征匹配器缓存
        self.current_method = MatchMethod.COSINE
        self.use_deep_feature = use_deep_feature
        self.segment_method = segment_method
        
        # HOG 特征提取器（延迟加载）
        self._deep_extractor = None
        
        # 预加载常用类型
        self._get_matcher("140")
    
    def _get_deep_extractor(self):
        """延迟加载 HOG 特征提取器"""
        if self._deep_extractor is None:
            from .deep_feature import HOGFeatureExtractor
            self._deep_extractor = HOGFeatureExtractor()
        return self._deep_extractor
    
    def _get_matcher(self, plate_type: str):
        """获取或创建指定类型的匹配器"""
        if self.use_deep_feature:
            if plate_type not in self.deep_matchers:
                from .deep_feature import HOGTemplateMatcher
                self.deep_matchers[plate_type] = HOGTemplateMatcher(
                    self.template_dir, plate_type, self._get_deep_extractor()
                )
            return self.deep_matchers[plate_type]
        else:
            if plate_type not in self.matchers:
                self.matchers[plate_type] = TemplateMatcher(
                    self.template_dir, plate_type
                )
            return self.matchers[plate_type]
    
    def set_use_deep_feature(self, use_deep: bool):
        """切换是否使用 HOG 特征"""
        self.use_deep_feature = use_deep
    
    def set_segment_method(self, method: SegmentMethod):
        """
        设置分割方法
        
        Args:
            method: SegmentMethod 枚举值
        """
        self.segment_method = method
    
    def _segment_plate(self, plate_img: np.ndarray, expected_chars: int) -> List[np.ndarray]:
        """
        根据设置的方法进行字符分割
        
        Args:
            plate_img: 车牌图像
            expected_chars: 预期字符数量
            
        Returns:
            字符图像列表
        """
        # 使用 V3 带回退机制
        return segment_with_fallback(plate_img, expected_chars)
    
    def detect_plate_type(self, plate_img: np.ndarray) -> str:
        """
        根据车牌颜色判断类型
        
        Returns:
            "140": 蓝牌
            "220": 黄牌（大车，双行）
            "green": 绿牌（新能源）
        """
        if plate_img is None or plate_img.size == 0:
            return "140"
        
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        
        # 根据色调判断
        if 35 < h_mean < 77 and s_mean > 50:  # 绿色
            return "green"
        elif 11 < h_mean < 34:  # 黄色
            return "220"
        else:  # 蓝色或其他
            return "140"
    
    def recognize(self, plate_img: np.ndarray,
                  plate_type: str = None,
                  method: MatchMethod = None,
                  expected_chars: int = None) -> Tuple[str, List[Tuple[str, float]]]:
        """
        识别车牌
        
        Args:
            plate_img: 矫正后的车牌图像 (BGR)
            plate_type: 车牌类型，None则自动检测
            method: 匹配方法，None则使用默认
            expected_chars: 预期字符数，None则自动判断
        
        Returns:
            (识别结果字符串, [(字符, 置信度), ...])
        """
        if plate_img is None or plate_img.size == 0:
            return "", []
        
        # 自动检测车牌类型
        if plate_type is None:
            plate_type = self.detect_plate_type(plate_img)
        
        # 确定字符数
        if expected_chars is None:
            expected_chars = 8 if plate_type == "green" else 7
        
        # 获取匹配器
        matcher = self._get_matcher(plate_type)
        
        # 使用指定或默认方法
        match_method = method if method else self.current_method
        
        # 字符分割（使用配置的分割方法）
        char_images = self._segment_plate(plate_img, expected_chars)
        
        if len(char_images) == 0:
            return "", []
        
        # 逐个字符识别
        results = []
        recognized_chars = []
        
        for i, char_img in enumerate(char_images):
            # 第一个字符是汉字（省份简称）
            is_chinese = (i == 0)
            
            # 根据匹配器类型调用不同方法
            if self.use_deep_feature:
                # HOG 特征匹配，传递位置信息用于规则约束
                method_str = "cosine" if match_method == MatchMethod.COSINE else "euclidean"
                matches = matcher.match_character(
                    char_img, 
                    is_chinese=is_chinese,
                    method=method_str,
                    top_k=1,
                    position=i  # 传递位置信息
                )
            else:
                # 像素匹配
                matches = matcher.match_character(
                    char_img, 
                    is_chinese=is_chinese,
                    method=match_method,
                    top_k=1
                )
            
            if matches:
                char, score = matches[0]
                recognized_chars.append(char)
                results.append((char, score))
            else:
                recognized_chars.append('?')
                results.append(('?', 0.0))
        
        return ''.join(recognized_chars), results
    
    def recognize_with_details(self, plate_img: np.ndarray,
                               plate_type: str = None) -> dict:
        """
        识别车牌并返回详细信息
        
        Returns:
            {
                'plate_number': str,
                'plate_type': str,
                'confidence': float,
                'char_details': [(char, score), ...],
                'char_images': [np.ndarray, ...],
                'use_deep_feature': bool
            }
        """
        if plate_type is None:
            plate_type = self.detect_plate_type(plate_img)
        
        expected_chars = 8 if plate_type == "green" else 7
        char_images = self._segment_plate(plate_img, expected_chars)
        
        plate_number, char_details = self.recognize(
            plate_img, plate_type, expected_chars=expected_chars
        )
        
        # 计算整体置信度（各字符置信度的平均值）
        if char_details:
            avg_confidence = sum(score for _, score in char_details) / len(char_details)
        else:
            avg_confidence = 0.0
        
        return {
            'plate_number': plate_number,
            'plate_type': plate_type,
            'confidence': avg_confidence,
            'char_details': char_details,
            'char_images': char_images,
            'use_deep_feature': self.use_deep_feature
        }
    
    def set_match_method(self, method: str):
        """
        设置匹配方法
        
        Args:
            method: "euclidean", "cosine", 或 "correlation"
        """
        method_map = {
            "euclidean": MatchMethod.EUCLIDEAN,
            "cosine": MatchMethod.COSINE,
            "correlation": MatchMethod.CORRELATION
        }
        if method.lower() in method_map:
            self.current_method = method_map[method.lower()]


def test_recognizer():
    """测试函数"""
    import sys
    
    # 创建识别器，使用 V3 连通区域分析分割
    recognizer = TemplateRecognizer(segment_method=SegmentMethod.V3_FALLBACK)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # 测试不同分割方法
        print("=" * 50)
        for method in [SegmentMethod.V2_HYBRID, SegmentMethod.V3_CONNECTED, SegmentMethod.V3_FALLBACK]:
            recognizer.set_segment_method(method)
            result = recognizer.recognize_with_details(img)
            
            print(f"\n分割方法: {method.value}")
            print(f"车牌号: {result['plate_number']}")
            print(f"整体置信度: {result['confidence']:.2%}")
            print(f"分割字符数: {len(result['char_images'])}")
        print("=" * 50)
    else:
        print("用法: python recognizer.py <车牌图片路径>")
        print("\n支持的分割方法:")
        for method in SegmentMethod:
            print(f"  - {method.value}: {method.name}")


if __name__ == "__main__":
    test_recognizer()
