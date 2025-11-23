import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def convert_ccpd_to_yolo_pose(ccpd_root, save_root, train_ratio=0.8, ccpd_green_root=None):
    """
    将CCPD数据集转换为YOLOv8-Pose格式 (用于关键点检测)
    
    Args:
        ccpd_root (str): CCPD数据集根目录
        save_root (str): 保存YOLO格式数据集的根目录
        train_ratio (float): 训练集比例
        ccpd_green_root (str): CCPD绿牌数据集根目录 (可选)
    """
    
    # 创建目录结构
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        Path(save_root).joinpath(d).mkdir(parents=True, exist_ok=True)
        
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    def get_files(root):
        files = []
        if not root or not os.path.exists(root): return files
        print(f"正在扫描目录: {root} ...")
        for r, _, fs in os.walk(root):
            for f in fs:
                if any(f.lower().endswith(ext) for ext in image_extensions):
                    files.append(os.path.join(r, f))
        return files

    blue_files = get_files(ccpd_root)
    green_files = get_files(ccpd_green_root)
    
    print(f"找到蓝色车牌: {len(blue_files)} 张")
    print(f"找到绿色车牌: {len(green_files)} 张")
    
    image_files = blue_files + green_files
    print(f"共找到 {len(image_files)} 张图片")
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    def process_files(files, subset):
        for file_path in tqdm(files, desc=f"处理 {subset} 集"):
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                if len(parts) < 4:
                    continue
                
                # 1. 解析 bbox (index 2)
                bbox_str = parts[2] # min_x&min_y_max_x&max_y
                coords = bbox_str.split('_')
                p1 = coords[0].split('&')
                p2 = coords[1].split('&')
                x1_box, y1_box = int(p1[0]), int(p1[1])
                x2_box, y2_box = int(p2[0]), int(p2[1])
                
                # 2. 解析 vertices (index 3)
                # CCPD顺序: BR, BL, TL, TR
                # 格式: x1&y1_x2&y2_x3&y3_x4&y4
                vertices_str = parts[3]
                v_coords = vertices_str.split('_')
                
                # 解析四个点
                pts = []
                for v in v_coords:
                    vx, vy = v.split('&')
                    pts.append((int(vx), int(vy)))
                
                # CCPD原始顺序: 0:BR, 1:BL, 2:TL, 3:TR
                # 目标顺序 (YOLO Pose): TL, TR, BR, BL (顺时针或Z字形都可以，这里我们定义为 TL, TR, BR, BL)
                # 对应关系:
                # Target TL = CCPD[2]
                # Target TR = CCPD[3]
                # Target BR = CCPD[0]
                # Target BL = CCPD[1]
                
                keypoints = [pts[2], pts[3], pts[0], pts[1]]
                
                # 读取图片获取尺寸
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                height, width = img.shape[:2]
                
                # 归一化 bbox
                box_w = x2_box - x1_box
                box_h = y2_box - y1_box
                box_x = x1_box + box_w / 2
                box_y = y1_box + box_h / 2
                
                norm_box_x = box_x / width
                norm_box_y = box_y / height
                norm_box_w = box_w / width
                norm_box_h = box_h / height
                
                # 归一化 keypoints
                # 格式: x y visibility (2=visible)
                kpt_str_list = []
                for kpt in keypoints:
                    kx, ky = kpt
                    nkx = kx / width
                    nky = ky / height
                    kpt_str_list.append(f"{nkx:.6f} {nky:.6f} 2")
                
                kpt_line = " ".join(kpt_str_list)
                
                # 写入label
                # class_id x y w h kpt1_x kpt1_y kpt1_v ...
                label_filename = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(save_root, 'labels', subset, label_filename)
                
                with open(label_path, 'w') as f:
                    f.write(f"0 {norm_box_x:.6f} {norm_box_y:.6f} {norm_box_w:.6f} {norm_box_h:.6f} {kpt_line}\n")
                
                # 复制图片
                dst_img_path = os.path.join(save_root, 'images', subset, filename)
                shutil.copy2(file_path, dst_img_path)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    process_files(train_files, 'train')
    process_files(val_files, 'val')
    print(f"Pose数据集转换完成: {save_root}")

if __name__ == "__main__":
    CCPD_ROOT = "ccpd_base"
    CCPD_GREEN_ROOT = "ccpd_green" # 假设用户有这个文件夹
    SAVE_ROOT = "ccpd_pose_dataset"
    
    if os.path.exists(CCPD_ROOT):
        convert_ccpd_to_yolo_pose(CCPD_ROOT, SAVE_ROOT, train_ratio=0.8, ccpd_green_root=CCPD_GREEN_ROOT)
    else:
        print(f"请确保数据集存在于 {CCPD_ROOT}")
