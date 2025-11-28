"""
CNN 字符分类识别模块
使用 ResNet18 进行字符分类识别
基于 License-Plate-Recognition 项目的方法
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import cv2
import os
from typing import List, Optional


# 数据预处理（与训练时一致）
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 单通道扩展为三通道
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 中文字符分类器
class ChineseCharClassifier(nn.Module):
    def __init__(self, num_classes: int = 31):
        super(ChineseCharClassifier, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 英文数字字符分类器
class EngNumCharClassifier(nn.Module):
    def __init__(self, num_classes: int = 34):
        super(EngNumCharClassifier, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 类别映射
CHINESE_CLASS_MAP = [
    '川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', 
    '闽', '宁', '青', '琼', '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙' 
]

ENG_NUM_CLASS_MAP = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]


class CNNCharRecognizer:
    """
    基于 CNN 的字符识别器
    使用 ResNet18 分类器识别字符
    """
    
    def __init__(self, checkpoint_dir: str = None):
        """
        初始化 CNN 字符识别器
        
        Args:
            checkpoint_dir: 模型权重目录路径
        """
        # 设置设备
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"CNN 识别器使用设备: {self.device}")
        
        # 默认模型目录
        if checkpoint_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_dir = os.path.join(base_dir, "checkpoints")
        
        self.checkpoint_dir = checkpoint_dir
        
        # 模型
        self.chinese_model: Optional[ChineseCharClassifier] = None
        self.eng_num_model: Optional[EngNumCharClassifier] = None
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载预训练模型"""
        # 中文字符模型
        chinese_paths = [
            os.path.join(self.checkpoint_dir, "chinese_char_classifier_0.9879.pth"),
            os.path.join(self.checkpoint_dir, "chinese_char_classifier_0.9545.pth"),
            os.path.join(self.checkpoint_dir, "chinese_char_classifier.pth"),
        ]
        
        for path in chinese_paths:
            if os.path.exists(path):
                print(f"加载中文模型: {os.path.basename(path)}")
                self.chinese_model = ChineseCharClassifier(num_classes=31)
                self.chinese_model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )
                self.chinese_model.to(self.device)
                self.chinese_model.eval()
                break
        
        if self.chinese_model is None:
            print("警告: 未找到中文字符模型")
        
        # 英文数字模型（34类：0-9 + A-Z去掉I和O）
        eng_paths = [
            os.path.join(self.checkpoint_dir, "char_classifier_0.9904.pth"),  # 34类
            os.path.join(self.checkpoint_dir, "eng_num_char_classifier.pth"),
        ]
        
        for path in eng_paths:
            if os.path.exists(path):
                print(f"加载英文数字模型: {os.path.basename(path)}")
                self.eng_num_model = EngNumCharClassifier(num_classes=34)
                self.eng_num_model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )
                self.eng_num_model.to(self.device)
                self.eng_num_model.eval()
                break
        
        if self.eng_num_model is None:
            print("警告: 未找到英文数字模型")
    
    def preprocess_image(self, image) -> torch.Tensor:
        """
        预处理图像
        
        LPR 分割算法已经输出黑底白字的二值图像，
        所以这里只需要简单处理。
        
        Args:
            image: PIL Image 或 numpy.ndarray (黑底白字)
        
        Returns:
            预处理后的张量
        """
        if isinstance(image, np.ndarray):
            # 确保是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 检查是否需要反转（确保黑底白字）
            # 黑底白字的平均值应该较低（<127）
            mean_val = np.mean(gray)
            if mean_val > 127:
                gray = 255 - gray
            
            # 居中处理：去除边缘空白
            coords = cv2.findNonZero(gray)
            if coords is not None and len(coords) > 10:
                x, y, w_roi, h_roi = cv2.boundingRect(coords)
                
                if w_roi > 3 and h_roi > 3:
                    char_roi = gray[y:y+h_roi, x:x+w_roi]
                    
                    # 创建正方形画布
                    size = max(w_roi, h_roi) + 4
                    canvas = np.zeros((size, size), dtype=np.uint8)
                    
                    y_offset = (size - h_roi) // 2
                    x_offset = (size - w_roi) // 2
                    canvas[y_offset:y_offset+h_roi, x_offset:x_offset+w_roi] = char_roi
                    
                    gray = canvas
            
            image = Image.fromarray(gray)
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def recognize_single(self, char_image, is_chinese: bool = False) -> tuple:
        """
        识别单个字符
        
        Args:
            char_image: 字符图像
            is_chinese: 是否是中文字符
        
        Returns:
            (字符, 置信度)
        """
        tensor = self.preprocess_image(char_image)
        
        if is_chinese:
            if self.chinese_model is None:
                return ('?', 0.0)
            with torch.no_grad():
                output = self.chinese_model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                char = CHINESE_CLASS_MAP[predicted.item()]
                return (char, confidence.item())
        else:
            if self.eng_num_model is None:
                return ('?', 0.0)
            with torch.no_grad():
                output = self.eng_num_model(tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                char = ENG_NUM_CLASS_MAP[predicted.item()]
                return (char, confidence.item())
    
    def recognize_plate(self, char_images: List[np.ndarray]) -> tuple:
        """
        识别完整车牌
        
        Args:
            char_images: 分割后的字符图像列表
        
        Returns:
            (识别结果字符串, [(字符, 置信度), ...])
        """
        if len(char_images) < 2:
            return ("", [])
        
        results = []
        chars = []
        
        # 第一个字符是中文省份
        char, conf = self.recognize_single(char_images[0], is_chinese=True)
        chars.append(char)
        results.append((char, conf))
        
        # 后续字符是英文和数字
        for char_img in char_images[1:]:
            char, conf = self.recognize_single(char_img, is_chinese=False)
            chars.append(char)
            results.append((char, conf))
        
        plate_number = ''.join(chars)
        return (plate_number, results)
    
    def recognize_with_details(self, char_images: List[np.ndarray]) -> dict:
        """
        识别车牌并返回详细信息
        
        Returns:
            {
                'plate_number': str,
                'confidence': float,
                'char_details': [(char, score), ...],
                'method': 'cnn_resnet18'
            }
        """
        plate_number, char_details = self.recognize_plate(char_images)
        
        if char_details:
            avg_confidence = sum(conf for _, conf in char_details) / len(char_details)
        else:
            avg_confidence = 0.0
        
        return {
            'plate_number': plate_number,
            'confidence': avg_confidence,
            'char_details': char_details,
            'method': 'cnn_resnet18'
        }


if __name__ == "__main__":
    # 测试代码
    recognizer = CNNCharRecognizer()
    print(f"中文模型: {'已加载' if recognizer.chinese_model else '未加载'}")
    print(f"英文数字模型: {'已加载' if recognizer.eng_num_model else '未加载'}")
