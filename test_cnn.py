"""
测试 CNN 字符识别的详细调试脚本
"""

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cnn_classifier import CNNCharRecognizer, transform, CHINESE_CLASS_MAP, ENG_NUM_CLASS_MAP
from template_match.segment_v3 import segment_with_fallback


def test_single_char(recognizer, char_img, is_chinese=False, expected_char=None):
    """测试单个字符识别，并显示 top-5 预测"""
    
    # 预处理
    if len(char_img.shape) == 3:
        char_img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        char_img_gray = char_img
    
    # 转为 PIL
    pil_img = Image.fromarray(char_img_gray)
    
    # 应用 transform
    tensor = transform(pil_img).unsqueeze(0).to(recognizer.device)
    
    # 选择模型
    if is_chinese:
        model = recognizer.chinese_model
        class_map = CHINESE_CLASS_MAP
    else:
        model = recognizer.eng_num_model
        class_map = ENG_NUM_CLASS_MAP
    
    if model is None:
        print("模型未加载!")
        return
    
    # 推理
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        
        # Top-5 预测
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
        
        print(f"\n{'中文' if is_chinese else '英文/数字'} 字符识别:")
        if expected_char:
            print(f"  期望: {expected_char}")
        print("  Top-5 预测:")
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_probs[0][i].item()
            char = class_map[idx]
            marker = " <--" if expected_char and char == expected_char else ""
            print(f"    {i+1}. {char} ({prob:.1%}){marker}")
    
    return class_map[top5_indices[0][0].item()]


def visualize_preprocessing(char_img):
    """可视化预处理过程"""
    
    # 原始图像
    if len(char_img.shape) == 3:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_img.copy()
    
    # PIL 图像
    pil_img = Image.fromarray(gray)
    
    # 应用 transform
    tensor = transform(pil_img)
    
    # 反归一化以便可视化
    tensor_vis = tensor * 0.5 + 0.5  # 反 Normalize
    tensor_np = tensor_vis[0].numpy()  # 取第一个通道
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title(f'原图 {gray.shape}')
    axes[0].axis('off')
    
    # Resize 后
    resized = cv2.resize(gray, (32, 32))
    axes[1].imshow(resized, cmap='gray')
    axes[1].set_title('Resize 32x32')
    axes[1].axis('off')
    
    # 归一化后
    axes[2].imshow(tensor_np, cmap='gray')
    axes[2].set_title('ToTensor + Normalize')
    axes[2].axis('off')
    
    # 直方图
    axes[3].hist(gray.flatten(), bins=50, alpha=0.7, label='原图')
    axes[3].hist(resized.flatten(), bins=50, alpha=0.7, label='32x32')
    axes[3].legend()
    axes[3].set_title('像素直方图')
    
    plt.tight_layout()
    plt.savefig('preprocess_debug.png', dpi=150)
    plt.show()


def test_with_template_chars():
    """使用模板字符测试模型"""
    
    recognizer = CNNCharRecognizer()
    
    # 测试中文模板
    chinese_dir = Path(__file__).parent / "character" / "chinese"
    if chinese_dir.exists():
        print("\n" + "="*50)
        print("测试中文模板字符:")
        print("="*50)
        
        for char_folder in sorted(chinese_dir.iterdir())[:5]:  # 测试前5个
            if char_folder.is_dir():
                expected_char = char_folder.name
                # 找一张图片
                for img_file in char_folder.glob("*.jpg"):
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        test_single_char(recognizer, img, is_chinese=True, expected_char=expected_char)
                        break
    
    # 测试英文数字模板
    alnum_dir = Path(__file__).parent / "character" / "alphanumeric"
    if alnum_dir.exists():
        print("\n" + "="*50)
        print("测试英文数字模板字符:")
        print("="*50)
        
        for char_folder in sorted(alnum_dir.iterdir())[:5]:  # 测试前5个
            if char_folder.is_dir():
                expected_char = char_folder.name
                for img_file in char_folder.glob("*.jpg"):
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        test_single_char(recognizer, img, is_chinese=False, expected_char=expected_char)
                        break


def test_from_plate(plate_img_path: str = None):
    """从车牌图片测试"""
    
    recognizer = CNNCharRecognizer()
    
    if plate_img_path:
        plate_img = cv2.imread(plate_img_path)
    else:
        # 使用 YOLO 检测
        from ultralytics import YOLO
        
        yolo_path = Path(__file__).parent / "ccpd_pose_runs" / "exp1" / "weights" / "best.pt"
        if not yolo_path.exists():
            print("未找到 YOLO 模型")
            return
        
        model = YOLO(str(yolo_path))
        
        # 查找测试图片
        test_imgs = list(Path(__file__).parent.glob("*.jpg"))
        if not test_imgs:
            print("未找到测试图片")
            return
        
        img = cv2.imread(str(test_imgs[0]))
        results = model(img, verbose=False)
        
        if not results or results[0].keypoints is None:
            print("未检测到车牌")
            return
        
        kpts = results[0].keypoints.xy.cpu().numpy()[0].astype(np.float32)
        
        dst_pts = np.array([[0, 0], [240, 0], [240, 80], [0, 80]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(kpts, dst_pts)
        plate_img = cv2.warpPerspective(img, M, (240, 80))
    
    # 分割
    chars = segment_with_fallback(plate_img, 7)
    print(f"\n分割出 {len(chars)} 个字符")
    
    # 显示分割结果
    fig, axes = plt.subplots(2, len(chars), figsize=(2*len(chars), 5))
    
    for i, char_img in enumerate(chars):
        # 显示字符
        axes[0, i].imshow(char_img, cmap='gray')
        axes[0, i].set_title(f'字符 {i+1}')
        axes[0, i].axis('off')
        
        # 显示预处理后
        pil_img = Image.fromarray(char_img)
        tensor = transform(pil_img)
        tensor_vis = (tensor * 0.5 + 0.5)[0].numpy()
        axes[1, i].imshow(tensor_vis, cmap='gray')
        axes[1, i].set_title('预处理后')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('char_preprocess.png', dpi=150)
    plt.show()
    
    # 识别每个字符
    print("\n识别结果:")
    for i, char_img in enumerate(chars):
        is_chinese = (i == 0)
        test_single_char(recognizer, char_img, is_chinese=is_chinese)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', action='store_true', help='测试模板字符')
    parser.add_argument('--plate', type=str, help='车牌图片路径')
    parser.add_argument('--preprocess', action='store_true', help='可视化预处理')
    
    args = parser.parse_args()
    
    if args.template:
        test_with_template_chars()
    elif args.plate:
        test_from_plate(args.plate)
    elif args.preprocess:
        # 随便读一个模板测试预处理
        template_dir = Path(__file__).parent / "character" / "alphanumeric" / "A"
        if template_dir.exists():
            for img_file in template_dir.glob("*.jpg"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    visualize_preprocessing(img)
                    break
    else:
        # 默认测试模板
        test_with_template_chars()
