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
    '1': ['I', 'L', '7'],
    'I': ['1', 'L'],
    'L': ['1', 'I'],
    '7': ['1', 'T'],
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
    'T': ['7', 'Y'],
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
    
    def normalize_char(self, img: np.ndarray) -> np.ndarray:
        """
        标准化字符图像（完整流程）：
        1. 灰度化
        2. 二值化（OTSU）
        3. 反色（统一为白底黑字）
        4. 去背景（取字符 bounding box）
        5. 等比例缩放
        6. 居中 + padding
        
        Args:
            img: BGR 或灰度图像
            
        Returns:
            标准化后的图像 (target_size)
        """
        H, W = self.target_size  # (64, 64) -> height=64, width=64
        
        # 1. 灰度化
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # 2. OTSU 二值化
        _, bin_img = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 反色保证"白底黑字"
        # 判断方式：如果图像平均值>127，说明白色多（背景是白色，字符是黑色）
        # 我们需要让字符是黑色(0)，背景是白色(255)
        # 如果当前白色像素多，说明已经是白底黑字，不需要反色
        if np.mean(bin_img) < 127:
            # 黑色像素多，说明是黑底白字，需要反色
            bin_img = 255 - bin_img
        
        # 4. 字符区域裁剪（找到字符的 bounding box）
        # 字符是黑色(0)，找非零像素需要反转
        char_pixels = cv2.findNonZero(255 - bin_img)
        
        if char_pixels is None or len(char_pixels) == 0:
            # 如果没有找到字符像素，返回白色画布
            return np.ones((H, W), dtype=np.uint8) * 255
        
        x, y, w, h = cv2.boundingRect(char_pixels)
        
        # 防止裁剪区域太小
        if w < 3 or h < 3:
            return np.ones((H, W), dtype=np.uint8) * 255
        
        char = bin_img[y:y+h, x:x+w]
        
        # 5. 等比例缩放（保留 4 像素边距）
        padding = 4
        scale = min((H - padding * 2) / h, (W - padding * 2) / w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 确保尺寸至少为1
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 6. 居中置入目标框
        canvas = np.ones((H, W), dtype=np.uint8) * 255
        start_x = (W - new_w) // 2
        start_y = (H - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas
    
    def preprocess_for_matching(self, img: np.ndarray) -> np.ndarray:
        """
        预处理图像以统一模板和输入的格式
        使用标准化流程：二值化 + 去背景 + 等比例缩放 + 居中
        
        Args:
            img: BGR 或灰度图像
        
        Returns:
            处理后的灰度图像（白底黑字，target_size 尺寸）
        """
        return self.normalize_char(img)
    
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
        
        # ========== 1 的特征检测（最关键）==========
        # 1 是非常细窄的竖线，宽高比极小，像素集中在中间竖直区域
        if char == '1':
            # 计算宽高比
            contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / (bh + 1e-6)
                
                # 1 的宽高比应该很小（细长）
                if aspect < 0.35:
                    score += 0.15
                elif aspect < 0.45:
                    score += 0.08
                
                # 1 的填充率应该较低（只有一条竖线）
                fill_ratio = cv2.contourArea(cnt) / (bw * bh + 1e-6)
                if fill_ratio < 0.5:
                    score += 0.1
            
            # 1 的中间区域像素分布：主要集中在中心垂直带
            mid_region = char_mask[:, w//3:2*w//3]
            side_region = np.concatenate([char_mask[:, :w//3], char_mask[:, 2*w//3:]], axis=1)
            
            mid_fill = np.sum(mid_region) / (mid_region.size * 255.0 + 1e-6)
            side_fill = np.sum(side_region) / (side_region.size * 255.0 + 1e-6)
            
            # 1 的特征：中间区域像素多，两侧像素少
            if mid_fill > side_fill * 2:
                score += 0.12
            elif mid_fill > side_fill * 1.5:
                score += 0.06
                
        # 0 的特征检测：是个闭合的椭圆/圆形
        elif char == '0':
            # 0 的中心应该是空的（闭合环形）
            center_region = char_mask[h//3:2*h//3, w//3:2*w//3]
            center_fill = np.sum(center_region) / (center_region.size * 255.0 + 1e-6)
            
            # 边缘应该有像素
            edge_fill = (np.sum(char_mask) - np.sum(center_region)) / ((char_mask.size - center_region.size) * 255.0 + 1e-6)
            
            # 0 的特征：中心空，边缘有像素
            if center_fill < 0.3 and edge_fill > 0.2:
                score += 0.12
                
            # 宽高比检测
            contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / (bh + 1e-6)
                # 0 的宽高比应该在 0.5-0.8 之间
                if 0.5 < aspect < 0.8:
                    score += 0.08
        
        # W 的特征检测：顶部宽，有两个 V 形谷
        elif char == 'W':
            # W 的宽高比应该较大（较宽）
            contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / (bh + 1e-6)
                
                # W 应该是宽的
                if aspect > 0.8:
                    score += 0.1
                    
            # W 的顶部应该有两个峰（两边高中间低）
            top_row = char_mask[:h//4, :]
            col_sums = np.sum(top_row, axis=0)
            # 找峰值
            if len(col_sums) > 4:
                left_peak = np.max(col_sums[:w//3])
                right_peak = np.max(col_sums[2*w//3:])
                center_val = np.max(col_sums[w//3:2*w//3])
                if left_peak > center_val and right_peak > center_val:
                    score += 0.1
        
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
        
        # 预处理图像，用于形状分析
        gray = self.extractor.preprocess_for_matching(char_img)
        h, w = gray.shape
        char_mask = 255 - gray  # 黑底白字
        
        # 计算基本形状特征（用于预筛选和加权）
        shape_info = self._analyze_shape(char_mask)
        
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
            
            # 基于形状特征的加权调整
            shape_adj = self._get_shape_adjustment(char, shape_info)
            score = score * (1.0 + shape_adj)
            
            scores.append((char, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 相似字符二次验证（只在前几名分数接近时）
        if len(scores) >= 2 and not is_chinese:
            top_char, top_score = scores[0]
            second_char, second_score = scores[1]
            
            # 分数差距小于 5%，进行二次验证
            if top_score - second_score < 0.05:
                if top_char in SIMILAR_CHARS and second_char in SIMILAR_CHARS.get(top_char, []):
                    # 使用结构特征进行调整
                    adj1 = self._get_structural_score(char_img, top_char)
                    adj2 = self._get_structural_score(char_img, second_char)
                    
                    new_score1 = top_score + adj1
                    new_score2 = second_score + adj2
                    
                    if new_score2 > new_score1:
                        scores[0], scores[1] = (second_char, new_score2), (top_char, new_score1)
            
            # 特殊处理：1 vs 0/W 的混淆（即使不在相似字符组中）
            if top_char in ['0', 'W', 'O'] and '1' in candidates:
                # 检查 1 是否在前几名中
                one_score = next((s for c, s in scores if c == '1'), 0)
                if one_score > 0:
                    # 使用形状特征判断
                    is_likely_one = self._is_likely_digit_one(char_mask)
                    if is_likely_one:
                        # 大幅提升 1 的分数
                        scores = [(c, s + 0.3 if c == '1' else s) for c, s in scores]
                        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def _analyze_shape(self, char_mask: np.ndarray) -> dict:
        """
        分析字符形状的基本特征
        """
        h, w = char_mask.shape
        info = {
            'aspect_ratio': 1.0,
            'fill_ratio': 0.5,
            'center_hollow': False,
            'is_narrow': False,
            'is_wide': False,
        }
        
        contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(cnt)
            info['aspect_ratio'] = bw / (bh + 1e-6)
            info['fill_ratio'] = cv2.contourArea(cnt) / (bw * bh + 1e-6)
            info['is_narrow'] = info['aspect_ratio'] < 0.4
            info['is_wide'] = info['aspect_ratio'] > 0.9
        
        # 检查中心是否空心
        center_region = char_mask[h//3:2*h//3, w//3:2*w//3]
        center_fill = np.sum(center_region) / (center_region.size * 255.0 + 1e-6)
        info['center_hollow'] = center_fill < 0.25
        
        return info
    
    def _get_shape_adjustment(self, char: str, shape_info: dict) -> float:
        """
        基于形状特征返回分数调整值
        """
        adj = 0.0
        
        # 1, I, L 应该是细窄的
        if char in ['1', 'I', 'L', '7'] and shape_info['is_narrow']:
            adj += 0.08
        
        # 0, O, D 应该有空心中心
        if char in ['0', 'O', 'D'] and shape_info['center_hollow']:
            adj += 0.05
        
        # W, M 应该是宽的
        if char in ['W', 'M'] and shape_info['is_wide']:
            adj += 0.05
        
        # 惩罚不匹配的形状
        if char in ['1', 'I'] and shape_info['is_wide']:
            adj -= 0.15  # 1 和 I 不应该是宽的
        
        if char in ['W', 'M'] and shape_info['is_narrow']:
            adj -= 0.15  # W 和 M 不应该是窄的
            
        if char in ['0', 'O', 'D'] and shape_info['is_narrow']:
            adj -= 0.1  # 0, O, D 不应该太窄
        
        return adj
    
    def _is_likely_digit_one(self, char_mask: np.ndarray) -> bool:
        """
        判断字符是否很可能是数字 1
        基于多个形状特征综合判断
        """
        h, w = char_mask.shape
        
        # 1. 检查宽高比（1 应该很窄）
        contours, _ = cv2.findContours(char_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
            
        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-6)
        
        # 宽高比太大，不是 1
        if aspect > 0.5:
            return False
        
        # 2. 检查像素分布（1 的像素应该集中在中间垂直带）
        left_third = char_mask[:, :w//3]
        middle_third = char_mask[:, w//3:2*w//3]
        right_third = char_mask[:, 2*w//3:]
        
        left_fill = np.sum(left_third) / (left_third.size + 1e-6)
        mid_fill = np.sum(middle_third) / (middle_third.size + 1e-6)
        right_fill = np.sum(right_third) / (right_third.size + 1e-6)
        
        # 中间区域像素应该明显多于两侧
        if mid_fill < left_fill + right_fill:
            return False
        
        # 3. 检查是否有空心（0 有空心，1 没有）
        center_region = char_mask[h//3:2*h//3, w//3:2*w//3]
        center_fill = np.sum(center_region) / (center_region.size * 255.0 + 1e-6)
        
        # 如果中心几乎是空的，更可能是 0
        if center_fill < 0.15:
            return False
        
        # 4. 填充率检查（1 的填充率应该较低）
        total_fill = np.sum(char_mask) / (char_mask.size * 255.0 + 1e-6)
        if total_fill > 0.4:
            return False
            
        return True
    
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
