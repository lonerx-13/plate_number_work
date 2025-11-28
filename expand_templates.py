"""
字符模板扩充工具
用于从车牌识别结果中提取字符样本，扩充模板库

使用方法:
1. 运行脚本，输入字符图像路径
2. 输入正确的字符标签
3. 脚本会自动保存为 A_1.png, A_2.png 等格式

示例:
    python expand_templates.py char_image.png A
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


def get_next_template_name(template_dir: str, char: str) -> str:
    """
    获取下一个模板文件名
    如果 A.png 存在，返回 A_1.png；如果 A_1.png 存在，返回 A_2.png，以此类推
    """
    base_path = os.path.join(template_dir, f"{char}.png")
    
    # 如果基础模板不存在，使用基础名称
    if not os.path.exists(base_path):
        return base_path
    
    # 查找下一个可用的编号
    idx = 1
    while True:
        path = os.path.join(template_dir, f"{char}_{idx}.png")
        if not os.path.exists(path):
            return path
        idx += 1


def preprocess_char_image(img: np.ndarray, target_size: tuple = (20, 20)) -> np.ndarray:
    """
    预处理字符图像为标准模板格式（白底黑字）
    
    处理流程:
    1. CLAHE 对比度增强（解决灰底灰字问题）
    2. Otsu 二值化
    3. 自动检测并转为白底黑字
    4. 去除边缘空白，居中放置
    
    Args:
        img: 输入图像
        target_size: 目标尺寸 (height, width)
        
    Returns:
        处理后的二值图像（白底黑字）
    """
    # 转灰度
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. CLAHE 对比度增强（解决灰底灰字的问题）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # 2. Otsu 二值化（自动找最佳阈值）
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 检查二值化效果，必要时使用自适应二值化
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio < 0.1 or white_ratio > 0.9:
        # 二值化效果不好，使用自适应二值化
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    # 4. 自动检测并确保是白底黑字
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
    
    # 5. 去除边缘空白，找到字符区域
    coords = cv2.findNonZero(255 - binary)  # 找黑色像素（字符）
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        # 添加一点边距
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(binary.shape[1] - x, bw + 2 * pad)
        bh = min(binary.shape[0] - y, bh + 2 * pad)
        binary = binary[y:y+bh, x:x+bw]
    
    # 6. 调整到目标尺寸（保持宽高比）
    h, w = binary.shape
    if h > 0 and w > 0:
        scale = min(target_size[0] / h, target_size[1] / w) * 0.8  # 留一些边距
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 创建目标大小的白色背景
            result = np.ones(target_size, dtype=np.uint8) * 255
            
            # 居中放置
            y_offset = (target_size[0] - new_h) // 2
            x_offset = (target_size[1] - new_w) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return result
    
    return cv2.resize(binary, target_size[::-1], interpolation=cv2.INTER_AREA)


def add_template(img_path: str, char_label: str, template_dir: str, 
                 is_chinese: bool = False) -> str:
    """
    添加新的字符模板
    
    Args:
        img_path: 字符图像路径
        char_label: 字符标签（如 'A', '京'）
        template_dir: 模板目录
        is_chinese: 是否是汉字
        
    Returns:
        保存的模板路径
    """
    # 读取图像
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    
    # 确定保存目录
    if is_chinese:
        save_dir = os.path.join(template_dir, "chinese")
    else:
        save_dir = os.path.join(template_dir, "alphanumeric")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 预处理
    processed = preprocess_char_image(img)
    
    # 获取文件名
    save_path = get_next_template_name(save_dir, char_label)
    
    # 保存
    cv2.imencode('.png', processed)[1].tofile(save_path)
    
    print(f"✓ 模板已保存: {save_path}")
    
    # 删除旧的缓存文件
    cache_path = os.path.splitext(save_path)[0] + "_hogv2.npy"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  已删除旧缓存: {cache_path}")
    
    return save_path


def interactive_mode(template_dir: str):
    """
    交互模式：逐个处理字符图像
    """
    print("=" * 50)
    print("字符模板扩充工具 - 交互模式")
    print("=" * 50)
    print(f"模板目录: {template_dir}")
    print("输入 'q' 退出")
    print()
    
    while True:
        img_path = input("请输入字符图像路径 (或 'q' 退出): ").strip()
        
        if img_path.lower() == 'q':
            print("退出")
            break
        
        if not os.path.exists(img_path):
            print(f"❌ 文件不存在: {img_path}")
            continue
        
        # 显示图像
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cv2.imshow("Character", img)
            cv2.waitKey(500)
        
        char_label = input("请输入正确的字符标签: ").strip()
        
        if not char_label:
            print("❌ 标签不能为空")
            continue
        
        # 判断是否是汉字
        is_chinese = len(char_label) == 1 and '\u4e00' <= char_label <= '\u9fff'
        
        try:
            add_template(img_path, char_label, template_dir, is_chinese)
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        print()
    
    cv2.destroyAllWindows()


def batch_mode(template_dir: str, char_dir: str, char_label: str):
    """
    批量模式：处理一个目录下的所有图像
    
    Args:
        template_dir: 模板目录
        char_dir: 字符图像目录
        char_label: 字符标签
    """
    is_chinese = len(char_label) == 1 and '\u4e00' <= char_label <= '\u9fff'
    
    count = 0
    for filename in os.listdir(char_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(char_dir, filename)
            try:
                add_template(img_path, char_label, template_dir, is_chinese)
                count += 1
            except Exception as e:
                print(f"❌ 处理失败 {filename}: {e}")
    
    print(f"\n共添加 {count} 个模板")


if __name__ == "__main__":
    # 默认模板目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, "character")
    
    if len(sys.argv) == 1:
        # 交互模式
        interactive_mode(template_dir)
    elif len(sys.argv) == 3:
        # 单个文件模式: python expand_templates.py image.png A
        img_path = sys.argv[1]
        char_label = sys.argv[2]
        is_chinese = len(char_label) == 1 and '\u4e00' <= char_label <= '\u9fff'
        add_template(img_path, char_label, template_dir, is_chinese)
    elif len(sys.argv) == 4 and sys.argv[1] == '--batch':
        # 批量模式: python expand_templates.py --batch dir/ A
        batch_mode(template_dir, sys.argv[2], sys.argv[3])
    else:
        print("用法:")
        print("  交互模式:  python expand_templates.py")
        print("  单文件:    python expand_templates.py image.png A")
        print("  批量模式:  python expand_templates.py --batch char_dir/ A")
