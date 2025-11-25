import cv2
import numpy as np
import os
from tqdm import tqdm

def augment_plate_color(img):
    """
    生成黄色和绿色车牌数据
    原理: 
    1. 蓝色车牌 (白字蓝底) -> 反转 -> 黄色车牌 (黑字黄底)
    2. 黄色车牌 -> 色相偏移 -> 绿色车牌 (黑字绿底)
    """
    # 1. 生成黄色车牌 (反转)
    # 蓝色 (B=255, G=0, R=0) -> 反转 -> (0, 255, 255) = 黄色
    # 白色 (255, 255, 255) -> 反转 -> (0, 0, 0) = 黑色
    img_yellow = cv2.bitwise_not(img)
    
    # 2. 生成绿色车牌 (从黄色偏移色相)
    # 黄色 HSV H约30 -> 绿色 HSV H约60
    hsv = cv2.cvtColor(img_yellow, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 黄色(30) -> 绿色(60-80)
    # 加 40 左右
    h_new = (h.astype(int) + 40) % 180
    h_new = h_new.astype(np.uint8)
    
    hsv_green = cv2.merge([h_new, s, v])
    img_green = cv2.cvtColor(hsv_green, cv2.COLOR_HSV2BGR)
    
    return img_yellow, img_green

def generate_augmented_data(src_dir, dst_dir):
    """
    读取src_dir中的蓝色车牌，生成黄/绿车牌保存到dst_dir
    """
    os.makedirs(os.path.join(dst_dir, 'yellow'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'green'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'blue'), exist_ok=True) # 原始图片也复制一份
    
    print(f"正在生成增强数据: {src_dir} -> {dst_dir}")
    
    for filename in tqdm(os.listdir(src_dir)):
        if not filename.endswith(('.jpg', '.png')):
            continue
            
        # 检查是否为绿牌 (根据车牌字符长度)
        # 文件名格式: 车牌号_原文件名.jpg
        # 蓝牌/黄牌通常7位，绿牌8位
        plate_str = filename.split('_')[0]
        if len(plate_str) > 7:
            # 已经是绿牌，直接复制到 color_dataset/green (如果尚未存在)
            # 但 ccpd_to_lpr.py 可能已经处理过了，这里为了保险起见，可以跳过或者复制
            # 我们假设 ccpd_to_lpr.py 已经处理了真实绿牌到 color_dataset/green
            # 所以这里直接跳过，避免把绿牌反转成紫色
            continue
            
        img_path = os.path.join(src_dir, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        
        # 保存原始蓝色到 color_dataset (用于ColorNet训练)
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(dst_dir, 'blue', filename))
        
        # 生成黄绿
        img_yellow, img_green = augment_plate_color(img)
        
        # 1. 保存到 color_dataset (用于ColorNet训练)
        cv2.imencode('.jpg', img_yellow)[1].tofile(os.path.join(dst_dir, 'yellow', f"yellow_{filename}"))
        # cv2.imencode('.jpg', img_green)[1].tofile(os.path.join(dst_dir, 'green', f"green_{filename}"))
        
        # 2. 保存回 lpr_dataset (用于LPRNet混合训练)
        # 保持标签在文件名前缀: Label_AugType_OrigName.jpg
        # 原文件名: Label_OrigName.jpg
        parts = filename.split('_')
        label = parts[0]
        rest = "_".join(parts[1:])
        
        # 黄色车牌 (黑字)
        yellow_name = f"{label}_yellow_{rest}"
        cv2.imencode('.jpg', img_yellow)[1].tofile(os.path.join(src_dir, yellow_name))
        
        # 绿色车牌 (黑字)
        # green_name = f"{label}_green_{rest}"
        # cv2.imencode('.jpg', img_green)[1].tofile(os.path.join(src_dir, green_name))

if __name__ == "__main__":
    # 假设你已经运行了 lprnet/ccpd_to_lpr.py 生成了 lpr_dataset
    LPR_ROOT = "./lpr_dataset"
    DST_DIR = "./color_dataset"
    
    for subset in ['train', 'val']:
        src_dir = os.path.join(LPR_ROOT, subset)
        if os.path.exists(src_dir):
            print(f"处理 {subset} 集...")
            generate_augmented_data(src_dir, DST_DIR)
        else:
            print(f"跳过 {subset} (目录不存在)")
            
    if not os.path.exists(LPR_ROOT):
        print(f"请先运行 lprnet/ccpd_to_lpr.py 生成基础数据")
