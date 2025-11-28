"""
模板匹配模块
使用归一化像素 + 欧氏距离/余弦相似度进行字符识别
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from enum import Enum


class MatchMethod(Enum):
    """匹配方法枚举"""
    EUCLIDEAN = "euclidean"      # 欧氏距离
    COSINE = "cosine"            # 余弦相似度
    CORRELATION = "correlation"  # 相关性匹配


class TemplateMatcher:
    """
    字符模板匹配器
    """
    
    # 标准化尺寸
    TEMPLATE_SIZE = (20, 40)  # (宽, 高)
    
    def __init__(self, template_dir: str, plate_type: str = "140"):
        """
        初始化模板匹配器
        
        Args:
            template_dir: 模板根目录 (包含 chinese/ 和 alphanumeric/ 子目录)
            plate_type: 车牌类型（目前统一使用相同模板，保留参数以兼容）
        """
        self.template_dir = template_dir
        self.plate_type = plate_type
        self.chinese_templates: Dict[str, np.ndarray] = {}
        self.alphanum_templates: Dict[str, np.ndarray] = {}
        
        self._load_templates()
    
    def _load_templates(self):
        """加载所有模板图像（新格式：直接以字符命名的 .png 文件）"""
        chinese_dir = os.path.join(self.template_dir, "chinese")
        alphanum_dir = os.path.join(self.template_dir, "alphanumeric")
        
        # 加载汉字模板
        if os.path.exists(chinese_dir):
            for filename in os.listdir(chinese_dir):
                if filename.endswith('.png'):
                    # 新格式：直接是字符名，如 "京.png" -> "京"
                    char = filename.replace('.png', '')
                    filepath = os.path.join(chinese_dir, filename)
                    template = self._load_and_normalize(filepath)
                    if template is not None:
                        self.chinese_templates[char] = template
        
        # 加载字母数字模板
        if os.path.exists(alphanum_dir):
            for filename in os.listdir(alphanum_dir):
                if filename.endswith('.png'):
                    # 新格式：直接是字符名，如 "A.png" -> "A"
                    char = filename.replace('.png', '')
                    filepath = os.path.join(alphanum_dir, filename)
                    template = self._load_and_normalize(filepath)
                    if template is not None:
                        self.alphanum_templates[char] = template
        
        print(f"加载模板: {len(self.chinese_templates)} 个汉字, "
              f"{len(self.alphanum_templates)} 个字母数字")
    
    def _load_and_normalize(self, filepath: str) -> Optional[np.ndarray]:
        """
        加载并归一化模板图像
        """
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # 调整到标准尺寸
        img = cv2.resize(img, self.TEMPLATE_SIZE)
        
        # 二值化处理（增强字符边缘，提高区分度）
        # Otsu 自动阈值
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 归一化到 [0, 1]
        img = binary.astype(np.float32) / 255.0
        
        return img
    
    def normalize_char_image(self, char_img: np.ndarray) -> np.ndarray:
        """
        归一化待识别的字符图像
        自动检测并处理反色问题（确保与模板一致：白底黑字）
        使用二值化增强字符形状特征
        """
        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        
        # 调整到标准尺寸
        char_img = cv2.resize(char_img, self.TEMPLATE_SIZE)
        
        # 二值化处理（Otsu 自动阈值）
        _, binary = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 归一化到 [0, 1]
        char_img = binary.astype(np.float32) / 255.0
        
        # 自动检测并处理反色
        # 模板是白底黑字（背景=1，字符=0）
        # 判断方法：计算边缘像素的平均值和中心区域像素的对比
        h, w = char_img.shape
        
        # 边缘区域（边框2像素）
        edge_pixels = np.concatenate([
            char_img[:2, :].flatten(),      # 上边
            char_img[-2:, :].flatten(),     # 下边
            char_img[:, :2].flatten(),      # 左边
            char_img[:, -2:].flatten()      # 右边
        ])
        edge_mean = np.mean(edge_pixels)
        
        # 中心区域
        center = char_img[h//4:3*h//4, w//4:3*w//4]
        center_mean = np.mean(center)
        
        # 如果边缘比中心更暗，说明是黑底白字，需要反色
        # （模板是白底黑字，边缘应该更亮）
        if edge_mean < center_mean - 0.1:
            char_img = 1.0 - char_img
        
        return char_img
    
    def euclidean_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算欧氏距离（越小越相似）
        """
        diff = img1.flatten() - img2.flatten()
        return np.sqrt(np.sum(diff ** 2))
    
    def cosine_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算余弦相似度（越大越相似，范围[-1, 1]）
        对于二值图像，使用改进的方法：只计算字符区域（黑色像素）
        """
        # 二值化后，字符是黑色(0)，背景是白色(1)
        # 提取字符掩码（值 < 0.5 的像素）
        mask1 = (img1 < 0.5).astype(np.float32)
        mask2 = (img2 < 0.5).astype(np.float32)
        
        # 计算交集和并集（类似 IoU）
        intersection = np.sum(mask1 * mask2)
        union = np.sum(np.maximum(mask1, mask2))
        
        if union == 0:
            return 0.0
        
        # 返回 IoU（交并比）作为相似度
        return intersection / union
    
    def hamming_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        计算汉明相似度（适合二值图像）
        统计相同像素的比例
        """
        # 二值化
        bin1 = (img1 > 0.5).astype(np.uint8)
        bin2 = (img2 > 0.5).astype(np.uint8)
        
        # 计算相同像素比例
        same = np.sum(bin1 == bin2)
        total = bin1.size
        
        return same / total
    
    def correlation_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        使用OpenCV模板匹配（相关性方法）
        """
        result = cv2.matchTemplate(
            (img1 * 255).astype(np.uint8), 
            (img2 * 255).astype(np.uint8), 
            cv2.TM_CCOEFF_NORMED
        )
        return result[0, 0]
    
    def match_character(self, char_img: np.ndarray, 
                        is_chinese: bool = False,
                        method: MatchMethod = MatchMethod.COSINE,
                        top_k: int = 1) -> List[Tuple[str, float]]:
        """
        匹配单个字符
        
        Args:
            char_img: 待识别的字符图像
            is_chinese: 是否匹配汉字（第一个字符）
            method: 匹配方法
            top_k: 返回前k个最佳匹配
        
        Returns:
            List of (字符, 相似度分数)，按相似度降序排列
        """
        # 归一化输入图像
        normalized = self.normalize_char_image(char_img)
        
        # 选择模板库
        templates = self.chinese_templates if is_chinese else self.alphanum_templates
        
        if not templates:
            return [('?', 0.0)]
        
        # 计算与所有模板的相似度
        scores = []
        for char, template in templates.items():
            if method == MatchMethod.EUCLIDEAN:
                # 欧氏距离（越小越好，转换为相似度）
                dist = self.euclidean_distance(normalized, template)
                score = 1.0 / (1.0 + dist)  # 转换为相似度
            elif method == MatchMethod.COSINE:
                score = self.cosine_similarity(normalized, template)
            else:  # CORRELATION
                score = self.correlation_match(normalized, template)
            
            scores.append((char, score))
        
        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def set_plate_type(self, plate_type: str):
        """
        切换车牌类型（新版本使用统一模板，此方法保留以兼容旧代码）
        """
        self.plate_type = plate_type
        # 新结构不需要重新加载，所有车牌类型使用相同模板


if __name__ == "__main__":
    # 测试代码
    import sys
    
    # 假设从项目根目录运行
    template_dir = "character"
    matcher = TemplateMatcher(template_dir)
    
    print(f"汉字模板: {list(matcher.chinese_templates.keys())}")
    print(f"字母数字模板: {list(matcher.alphanum_templates.keys())}")
