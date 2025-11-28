"""
基于 HOG 的特征提取器
用于字符模板匹配的特征提取
"""

import numpy as np
import cv2
import os
from typing import Optional, Dict, List, Tuple
from skimage.feature import hog


# 相似字符组定义
SIMILAR_CHARS = {
    '0': ['O', 'D'],
    'O': ['0', 'D'],
    'D': ['0', 'O'],
    '1': ['I', 'L'],
    'I': ['1', 'L'],
    'L': ['1', 'I'],
    '2': ['Z'],
    'Z': ['2'],
    '5': ['S'],
    'S': ['5'],
    '6': ['G'],
    'G': ['6'],
    '8': ['B'],
    'B': ['8'],
    'A': ['H', '4'],
    'H': ['A', 'N'],
    '4': ['A'],
    'N': ['H', 'M'],
    'M': ['N', 'W'],
    'W': ['M'],
    'C': ['G'],
    'E': ['F'],
    'F': ['E'],
    'P': ['R'],
    'R': ['P'],
    'K': ['X'],
    'X': ['K'],
    'U': ['V'],
    'V': ['U', 'Y'],
    'Y': ['V'],
}


class HOGFeatureExtractor:
    """
    使用 HOG 提取图像特征
    简单可靠的特征提取方案
    """
    
    def __init__(self, 
                 target_size: tuple = (64, 64),
                 orientations: int = 9,
                 pixels_per_cell: tuple = (8, 8),
                 cells_per_block: tuple = (2, 2)):
        """
        初始化 HOG 特征提取器
        
        Args:
            target_size: 统一的图像尺寸 (height, width)
            orientations: 梯度方向数量
            pixels_per_cell: 每个 cell 的像素数
            cells_per_block: 每个 block 包含的 cell 数
        """
        self.target_size = target_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        
        # 计算特征维度
        self._calculate_feature_dim()
        
        print(f"HOG 特征提取器已初始化")
        print(f"  - 目标尺寸: {target_size}")
        print(f"  - 特征维度: {self.feature_dim}")
    
    def _calculate_feature_dim(self):
        """计算 HOG 特征向量的维度"""
        temp_img = np.zeros(self.target_size, dtype=np.uint8)
        feat = hog(temp_img, 
                   orientations=self.orientations,
                   pixels_per_cell=self.pixels_per_cell,
                   cells_per_block=self.cells_per_block,
                   feature_vector=True)
        self.feature_dim = len(feat)
    
    def preprocess_for_matching(self, img: np.ndarray) -> np.ndarray:
        """
        预处理图像以统一模板和输入的格式
        将所有输入统一转换为高对比度的白底黑字图像
        
        Args:
            img: BGR 或灰度图像
        
        Returns:
            处理后的灰度图像（白底黑字，高对比度）
        """
        # 转灰度
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 1. 先进行对比度增强（解决灰底灰字的问题）
        # 使用 CLAHE（对比度受限的自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)
        
        # 2. Otsu 二值化（自动找最佳阈值）
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 如果 Otsu 效果不好（可能是图像本身对比度太低），使用自适应二值化
        # 检查二值化后的白色像素比例
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio < 0.1 or white_ratio > 0.9:
            # 二值化效果不好，使用自适应二值化
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        # 4. 自动检测并修正反色：确保是白底黑字
        h, w = binary.shape
        # 计算边缘区域的平均值
        edge_pixels = np.concatenate([
            binary[:max(2, h//10), :].flatten(),      # 上边缘
            binary[-max(2, h//10):, :].flatten(),     # 下边缘
            binary[:, :max(2, w//10)].flatten(),      # 左边缘
            binary[:, -max(2, w//10):].flatten()      # 右边缘
        ])
        edge_mean = np.mean(edge_pixels)
        
        # 计算中心区域的平均值
        center = binary[h//4:3*h//4, w//4:3*w//4]
        center_mean = np.mean(center)
        
        # 如果边缘比中心暗（黑底白字），需要反色
        if edge_mean < center_mean - 10:
            binary = 255 - binary
        
        # 5. 调整到目标尺寸
        resized = cv2.resize(binary, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_AREA)
        
        return resized
        
        return resized
    
    def extract_from_numpy(self, img: np.ndarray, preprocess: bool = True) -> np.ndarray:
        """
        从 numpy 数组提取 HOG 特征
        
        Args:
            img: BGR 或灰度图像
            preprocess: 是否进行预处理
        
        Returns:
            特征向量
        """
        if preprocess:
            gray = self.preprocess_for_matching(img)
        else:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            gray = cv2.resize(gray, (self.target_size[1], self.target_size[0]), 
                             interpolation=cv2.INTER_AREA)
        
        # 提取 HOG 特征
        feat = hog(gray, 
                   orientations=self.orientations,
                   pixels_per_cell=self.pixels_per_cell,
                   cells_per_block=self.cells_per_block,
                   feature_vector=True)
        
        return feat.astype(np.float32)
    
    def extract_from_file(self, filepath: str) -> Optional[np.ndarray]:
        """从文件提取特征"""
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return self.extract_from_numpy(img)
    
    def batch_extract(self, images: list) -> np.ndarray:
        """批量提取特征"""
        if len(images) == 0:
            return np.array([])
        
        features = []
        for img in images:
            feat = self.extract_from_numpy(img)
            features.append(feat)
        
        return np.array(features, dtype=np.float32)


class HOGTemplateMatcher:
    """
    基于 HOG 特征的模板匹配器
    """
    
    # 车牌中不使用的字母
    EXCLUDED_LETTERS = {'I', 'O'}
    
    def __init__(self, template_dir: str, plate_type: str = "140", 
                 feature_extractor: HOGFeatureExtractor = None):
        """
        初始化模板匹配器
        
        Args:
            template_dir: 模板根目录
            plate_type: 车牌类型
            feature_extractor: 共享的特征提取器
        """
        self.template_dir = template_dir
        self.plate_type = plate_type
        
        if feature_extractor is None:
            self.extractor = HOGFeatureExtractor()
        else:
            self.extractor = feature_extractor
        
        # 模板特征缓存
        self.chinese_features: Dict[str, np.ndarray] = {}
        self.alphanum_features: Dict[str, np.ndarray] = {}
        self.digit_features: Dict[str, np.ndarray] = {}
        self.letter_features: Dict[str, np.ndarray] = {}
        
        self._load_template_features()
    
    # 支持的图片格式
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp')
    
    def _get_cache_path(self, template_path: str) -> str:
        """获取特征缓存文件路径"""
        return os.path.splitext(template_path)[0] + "_hogv2.npy"
    
    def _parse_char_name(self, filename: str) -> Optional[str]:
        """
        从文件名解析字符名称
        支持格式: A.png, A_1.png, A_2.jpg 等
        
        Args:
            filename: 文件名（不含路径）
            
        Returns:
            字符名称，如 'A'，无效则返回 None
        """
        # 检查是否是支持的图片格式
        lower_name = filename.lower()
        ext = None
        for fmt in self.SUPPORTED_FORMATS:
            if lower_name.endswith(fmt):
                ext = fmt
                break
        
        if ext is None:
            return None
        
        # 移除扩展名
        name = filename[:len(filename) - len(ext)]
        
        # 处理变体格式: A_1, A_2 等
        if '_' in name:
            parts = name.split('_')
            # 检查是否是变体格式（如 A_1, 京_2）
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            # 特殊处理缓存文件名（如 A_hogv2）
            if parts[-1] in ['hogv2', 'hog', 'resnet']:
                return None
        
        return name
    
    def _load_template_features(self):
        """
        加载所有模板的 HOG 特征
        支持多模板：每个字符可以有多个变体（A.png, A_1.jpg, A_2.bmp）
        匹配时取所有变体中的最高分
        """
        chinese_dir = os.path.join(self.template_dir, "chinese")
        alphanum_dir = os.path.join(self.template_dir, "alphanumeric")
        
        print(f"加载模板 HOG 特征（支持格式: {', '.join(self.SUPPORTED_FORMATS)}）...")
        
        # 用于存储多模板特征
        # 格式: {char: [feat1, feat2, ...]}
        self.chinese_multi_features: Dict[str, List[np.ndarray]] = {}
        self.alphanum_multi_features: Dict[str, List[np.ndarray]] = {}
        
        # 加载汉字模板
        if os.path.exists(chinese_dir):
            for filename in os.listdir(chinese_dir):
                char = self._parse_char_name(filename)
                if char is None:
                    continue
                    
                filepath = os.path.join(chinese_dir, filename)
                
                cache_path = self._get_cache_path(filepath)
                if os.path.exists(cache_path):
                    feat = np.load(cache_path)
                else:
                    feat = self.extractor.extract_from_file(filepath)
                    if feat is not None:
                        np.save(cache_path, feat)
                
                if feat is not None:
                    # 添加到多模板列表
                    if char not in self.chinese_multi_features:
                        self.chinese_multi_features[char] = []
                    self.chinese_multi_features[char].append(feat)
                    
                    # 兼容旧接口：使用第一个模板作为默认
                    if char not in self.chinese_features:
                        self.chinese_features[char] = feat
        
        # 加载字母数字模板
        if os.path.exists(alphanum_dir):
            for filename in os.listdir(alphanum_dir):
                char = self._parse_char_name(filename)
                if char is None:
                    continue
                
                filepath = os.path.join(alphanum_dir, filename)
                
                cache_path = self._get_cache_path(filepath)
                if os.path.exists(cache_path):
                    feat = np.load(cache_path)
                else:
                    feat = self.extractor.extract_from_file(filepath)
                    if feat is not None:
                        np.save(cache_path, feat)
                
                if feat is not None:
                    # 添加到多模板列表
                    if char not in self.alphanum_multi_features:
                        self.alphanum_multi_features[char] = []
                    self.alphanum_multi_features[char].append(feat)
                    
                    # 兼容旧接口
                    if char not in self.alphanum_features:
                        self.alphanum_features[char] = feat
                        if char.isdigit():
                            self.digit_features[char] = feat
                        elif char.isalpha() and len(char) == 1:
                            self.letter_features[char] = feat
        
        # 统计多模板数量
        multi_count = sum(1 for v in self.alphanum_multi_features.values() if len(v) > 1)
        print(f"加载完成: {len(self.chinese_features)} 汉字, "
              f"{len(self.alphanum_features)} 字母数字")
        if multi_count > 0:
            print(f"  - 其中 {multi_count} 个字符有多个模板变体")
    
    def cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def euclidean_distance(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算欧氏距离（转换为相似度）"""
        dist = np.linalg.norm(feat1 - feat2)
        return 1.0 / (1.0 + dist)
    
    def _get_structural_score(self, char_img: np.ndarray, char: str) -> float:
        """
        获取字符的结构特征分数，用于区分相似字符
        只在需要时调用，不影响正常匹配
        """
        # 预处理
        gray = self.extractor.preprocess_for_matching(char_img)
        h, w = gray.shape
        char_mask = 255 - gray  # 黑底白字
        
        score = 0.0
        
        # 2 vs Z: 斜线方向不同
        # 2: 斜线从右上往左下弯曲（中间区域左边有像素）
        # Z: 斜线从右上往左下直线（中间区域也是右边有像素，或左右均衡）
        if char in ['2', 'Z']:
            # 检查中间区域（40%-60%高度）的左右像素分布
            mid_region = char_mask[2*h//5:3*h//5, :]  # 40%-60% 高度
            
            left_half = mid_region[:, :w//2]
            right_half = mid_region[:, w//2:]
            
            left_fill = np.sum(left_half) / (left_half.size * 255.0 + 1e-6)
            right_fill = np.sum(right_half) / (right_half.size * 255.0 + 1e-6)
            
            # 2 的中间区域：斜线刚从右边过来，所以右边更多
            # Z 的中间区域：斜线已经到中间，所以左右差不多
            if char == '2':
                # 2 的特征：中间区域右边明显多于左边
                if right_fill > left_fill + 0.15:
                    score += 0.12
            else:  # Z
                # Z 的特征：中间区域左右差不多，或左边略多
                if abs(right_fill - left_fill) < 0.2:
                    score += 0.12
        
        # 0 vs O vs D: 0 更细长
        if char in ['0', 'O', 'D']:
            # 找轮廓
            contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / (bh + 1e-6)
                
                if char == '0' and aspect < 0.7:  # 0 更细长
                    score += 0.03
                elif char == 'O' and 0.7 <= aspect <= 1.0:  # O 更圆
                    score += 0.03
                elif char == 'D' and aspect > 0.6:  # D 较宽
                    score += 0.03
        
        # 8 vs B: 8 更对称
        if char in ['8', 'B']:
            left_half = char_mask[:, :w//2]
            right_half = np.fliplr(char_mask[:, w//2:])
            min_w = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_w]
            right_half = right_half[:, :min_w]
            symmetry = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            
            if char == '8' and symmetry > 0.7:
                score += 0.03
            elif char == 'B' and symmetry < 0.7:
                score += 0.03
        
        # A vs H: A 是三角形顶部尖，H 是两条平行竖线
        if char in ['A', 'H']:
            # 检查顶部区域：A 顶部窄，H 顶部宽
            top_region = char_mask[:h//4, :]
            top_fill = np.sum(top_region, axis=0) / (h//4 * 255.0 + 1e-6)
            
            # 找到顶部有像素的列范围
            top_cols = np.where(top_fill > 0.1)[0]
            if len(top_cols) > 0:
                top_width_ratio = (top_cols[-1] - top_cols[0] + 1) / w
                
                # A 的顶部窄（三角形尖端）
                if char == 'A' and top_width_ratio < 0.5:
                    score += 0.05
                # H 的顶部宽（两条竖线）
                elif char == 'H' and top_width_ratio > 0.6:
                    score += 0.05
            
            # 检查中间横线位置：A 的横线在中偏下，H 的横线在正中
            mid_start = h // 3
            mid_end = 2 * h // 3
            for row in range(mid_start, mid_end):
                row_fill = np.sum(char_mask[row, :]) / (w * 255.0)
                if row_fill > 0.5:  # 找到横线
                    relative_pos = (row - mid_start) / (mid_end - mid_start)
                    if char == 'A' and relative_pos > 0.5:  # A 横线偏下
                        score += 0.03
                    elif char == 'H' and relative_pos < 0.5:  # H 横线居中偏上
                        score += 0.03
                    break
        
        # 4 vs A: 
        # 4: 横线下方只有一条竖线（在右侧）
        # A: 横线下方有两条竖线（在两侧）
        if char in ['4', 'A']:
            # 找到横线的位置（最宽的水平填充行）
            row_fills = [np.sum(char_mask[r, :]) / (w * 255.0) for r in range(h)]
            max_fill_row = np.argmax(row_fills)
            
            # 只分析横线下方的区域
            if max_fill_row < h - 3:  # 确保横线不在最底部
                below_line = char_mask[max_fill_row + 2:, :]
                if below_line.size > 0:
                    bh, bw = below_line.shape
                    
                    # 计算每一列的填充率
                    col_fills = np.sum(below_line, axis=0) / (bh * 255.0 + 1e-6)
                    
                    # 找到有字符的列（填充率 > 0.3）
                    active_cols = np.where(col_fills > 0.3)[0]
                    
                    if len(active_cols) > 0:
                        # 计算活跃列的分布
                        col_range = active_cols[-1] - active_cols[0] + 1
                        col_center = (active_cols[0] + active_cols[-1]) / 2 / w
                        
                        if char == '4':
                            # 4 横线下方：只有右边有竖线，占据较窄范围，位置偏右
                            if col_range < w * 0.4 and col_center > 0.5:
                                score += 0.12
                        else:  # A
                            # A 横线下方：两边都有竖线，占据较宽范围
                            if col_range > w * 0.5:
                                score += 0.12
        
        return score
    
    def _compute_multi_template_score(self, feat: np.ndarray, char: str, 
                                       is_chinese: bool, method: str) -> float:
        """
        计算多模板匹配分数（取所有模板变体中的最高分）
        
        Args:
            feat: 输入图像的特征
            char: 字符名称
            is_chinese: 是否是汉字
            method: 相似度计算方法
            
        Returns:
            最高相似度分数
        """
        if is_chinese:
            templates = self.chinese_multi_features.get(char, [])
        else:
            templates = self.alphanum_multi_features.get(char, [])
        
        if not templates:
            return 0.0
        
        max_score = 0.0
        for template_feat in templates:
            if method == "cosine":
                score = self.cosine_similarity(feat, template_feat)
            else:
                score = self.euclidean_distance(feat, template_feat)
            max_score = max(max_score, score)
        
        return max_score
    
    def match_character(self, char_img: np.ndarray, 
                        is_chinese: bool = False,
                        method: str = "cosine",
                        top_k: int = 1,
                        position: int = -1,
                        prefer_digit: bool = None) -> list:
        """
        匹配单个字符（支持多模板匹配）
        
        Args:
            char_img: 字符图像
            is_chinese: 是否匹配汉字
            method: "cosine" 或 "euclidean"
            top_k: 返回前k个匹配
            position: 字符位置（用于规则约束）
            prefer_digit: 是否优先数字
        
        Returns:
            [(字符, 相似度), ...]
        """
        # 提取特征
        feat = self.extractor.extract_from_numpy(char_img)
        
        # 确定候选字符集
        if is_chinese:
            candidates = set(self.chinese_features.keys())
        elif prefer_digit is True:
            candidates = set(self.digit_features.keys())
        elif prefer_digit is False:
            candidates = {k for k in self.letter_features.keys() 
                         if k not in self.EXCLUDED_LETTERS}
        else:
            # 根据位置选择
            if position == 1:
                # 第二位必须是字母
                candidates = {k for k in self.letter_features.keys() 
                             if k not in self.EXCLUDED_LETTERS}
            elif position >= 2:
                # 排除 I/O
                candidates = {k for k in self.alphanum_features.keys() 
                             if k not in self.EXCLUDED_LETTERS}
            else:
                candidates = set(self.alphanum_features.keys())
        
        if not candidates:
            return [('?', 0.0)]
        
        # 计算相似度（使用多模板匹配）
        scores = []
        for char in candidates:
            # 使用多模板匹配：取该字符所有模板变体中的最高分
            score = self._compute_multi_template_score(feat, char, is_chinese, method)
            scores.append((char, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 相似字符二次验证（只在前两名分数接近时）
        if len(scores) >= 2 and not is_chinese:
            top_char, top_score = scores[0]
            second_char, second_score = scores[1]
            
            # 分数差距小于 3%，且是相似字符
            if top_score - second_score < 0.03:
                if top_char in SIMILAR_CHARS and second_char in SIMILAR_CHARS.get(top_char, []):
                    # 使用结构特征进行调整
                    adj1 = self._get_structural_score(char_img, top_char)
                    adj2 = self._get_structural_score(char_img, second_char)
                    
                    new_score1 = top_score + adj1
                    new_score2 = second_score + adj2
                    
                    if new_score2 > new_score1:
                        scores[0], scores[1] = (second_char, new_score2), (top_char, new_score1)
        
        return scores[:top_k]
    
    def set_plate_type(self, plate_type: str):
        """切换车牌类型"""
        self.plate_type = plate_type


# 向后兼容别名
DeepTemplateMatcher = HOGTemplateMatcher
ResNetFeatureExtractor = HOGFeatureExtractor


if __name__ == "__main__":
    extractor = HOGFeatureExtractor()
    
    import sys
    if len(sys.argv) > 1:
        feat = extractor.extract_from_file(sys.argv[1])
        print(f"特征维度: {feat.shape}")
        print(f"特征范围: [{feat.min():.4f}, {feat.max():.4f}]")
