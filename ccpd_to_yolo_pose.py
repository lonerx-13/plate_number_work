import os
import shutil
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def convert_ccpd_to_yolo_pose(ccpd_root, save_root, train_ratio=0.8, ccpd_green_root=None, no_plate_root=None):
    """
    将CCPD数据集转换为YOLOv8-Pose格式 (用于关键点检测)
    支持混合蓝牌、绿牌以及无车牌负样本数据。
    
    Args:
        ccpd_root (str): CCPD数据集根目录 (蓝牌)
        save_root (str): 保存YOLO格式数据集的根目录
        train_ratio (float): 训练集比例
        ccpd_green_root (str): CCPD绿牌数据集根目录 (可选)
        no_plate_root (str): 无车牌背景图片目录 (可选，用于负样本训练)
    """
    
    # 创建目录结构
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for d in dirs:
        Path(save_root).joinpath(d).mkdir(parents=True, exist_ok=True)
        
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
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
    no_plate_files = get_files(no_plate_root)
    
    print(f"找到蓝色车牌: {len(blue_files)} 张")
    print(f"找到绿色车牌: {len(green_files)} 张")
    print(f"找到无车牌背景: {len(no_plate_files)} 张")
    
    # 构建数据列表，包含类型标记
    # type='plate': 需要解析CCPD文件名
    # type='no_plate': 只需要复制图片并生成空标签
    all_items = []
    for f in blue_files + green_files:
        all_items.append({'path': f, 'type': 'plate'})
    for f in no_plate_files:
        all_items.append({'path': f, 'type': 'no_plate'})
        
    print(f"共找到 {len(all_items)} 张图片")
    random.shuffle(all_items)
    
    split_idx = int(len(all_items) * train_ratio)
    train_items = all_items[:split_idx]
    val_items = all_items[split_idx:]
    
    def process_files(items, subset):
        for item in tqdm(items, desc=f"处理 {subset} 集"):
            file_path = item['path']
            file_type = item['type']
            filename = os.path.basename(file_path)
            
            dst_img_path = os.path.join(save_root, 'images', subset, filename)
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(save_root, 'labels', subset, label_filename)
            
            # 处理无车牌图片 (负样本)
            if file_type == 'no_plate':
                try:
                    shutil.copy2(file_path, dst_img_path)
                    # 创建空标签文件，告诉YOLO这里没有目标
                    with open(label_path, 'w') as f:
                        pass 
                except Exception as e:
                    print(f"Error processing no_plate {filename}: {e}")
                continue

            # 处理有车牌图片 (CCPD格式)
            try:
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
                vertices_str = parts[3]
                v_coords = vertices_str.split('_')
                
                pts = []
                for v in v_coords:
                    vx, vy = v.split('&')
                    pts.append((int(vx), int(vy)))
                
                # 转换顺序为 YOLO Pose: TL, TR, BR, BL
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
                kpt_str_list = []
                for kpt in keypoints:
                    kx, ky = kpt
                    nkx = kx / width
                    nky = ky / height
                    kpt_str_list.append(f"{nkx:.6f} {nky:.6f} 2")
                
                kpt_line = " ".join(kpt_str_list)
                
                # 写入label
                with open(label_path, 'w') as f:
                    f.write(f"0 {norm_box_x:.6f} {norm_box_y:.6f} {norm_box_w:.6f} {norm_box_h:.6f} {kpt_line}\n")
                
                # 复制图片
                shutil.copy2(file_path, dst_img_path)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    process_files(train_items, 'train')
    process_files(val_items, 'val')
    print(f"Pose数据集转换完成: {save_root}")

if __name__ == "__main__":
    # 配置路径
    CCPD_ROOT = "ccpd_challenge"           # 蓝牌数据
    CCPD_GREEN_ROOT = "ccpd_green"   # 绿牌数据
    NO_PLATE_ROOT = "ccpd_np" # 无车牌负样本数据 (请将无车牌图片放入此文件夹)
    SAVE_ROOT = "ccpd_pose_dataset"
    
    # 检查至少有一个数据源存在
    if os.path.exists(CCPD_ROOT) or os.path.exists(CCPD_GREEN_ROOT) or os.path.exists(NO_PLATE_ROOT):
        convert_ccpd_to_yolo_pose(
            ccpd_root=CCPD_ROOT, 
            save_root=SAVE_ROOT, 
            train_ratio=0.8, 
            ccpd_green_root=CCPD_GREEN_ROOT,
            no_plate_root=NO_PLATE_ROOT
        )
    else:
        print(f"未找到任何数据集目录，请检查路径配置。")
