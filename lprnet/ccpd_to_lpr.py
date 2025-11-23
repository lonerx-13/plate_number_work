import os
import cv2
import numpy as np
from tqdm import tqdm

# CCPD 字符映射表
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def four_point_transform(image, pts):
    width = 94  # LPRNet 输入宽
    height = 24 # LPRNet 输入高
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def convert_ccpd_to_lpr(ccpd_root, save_root):
    os.makedirs(os.path.join(save_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)
    
    image_files = []
    for root, _, files in os.walk(ccpd_root):
        for file in files:
            if file.endswith('.jpg'):
                image_files.append(os.path.join(root, file))
    
    # 简单划分
    split = int(len(image_files) * 0.9)
    train_files = image_files[:split]
    val_files = image_files[split:]
    
    def process(files, subset):
        for file_path in tqdm(files, desc=f"Processing {subset}"):
            try:
                filename = os.path.basename(file_path)
                parts = filename.split('-')
                if len(parts) < 5: continue
                
                # 解析车牌号
                plate_indices = parts[4].split('_')
                plate_str = PROVINCES[int(plate_indices[0])] + ADS[int(plate_indices[1])]
                for idx in plate_indices[2:]:
                    plate_str += ADS[int(idx)]
                
                # 解析Vertices进行矫正
                vertices_str = parts[3]
                v_coords = vertices_str.split('_')
                pts = []
                for v in v_coords:
                    vx, vy = v.split('&')
                    pts.append([int(vx), int(vy)])
                
                # CCPD: BR, BL, TL, TR -> TL, TR, BR, BL
                pts = np.array([pts[2], pts[3], pts[0], pts[1]], dtype="float32")
                
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None: continue
                
                crop = four_point_transform(img, pts)
                
                # 保存: 文件夹/车牌号.jpg
                # 注意: 车牌号可能包含中文，Windows下文件名可能乱码，建议使用索引或确保编码
                # 这里直接用车牌号作为文件名
                save_name = f"{plate_str}_{os.path.splitext(filename)[0]}.jpg"
                save_path = os.path.join(save_root, subset, save_name)
                
                # cv2.imwrite不支持中文路径，使用imencode
                cv2.imencode('.jpg', crop)[1].tofile(save_path)
                
            except Exception as e:
                continue

    process(train_files, 'train')
    process(val_files, 'val')

if __name__ == "__main__":
    CCPD_ROOT = "./ccpd_green" # 假设在上级目录
    SAVE_ROOT = "lpr_dataset"
    if os.path.exists(CCPD_ROOT):
        convert_ccpd_to_lpr(CCPD_ROOT, SAVE_ROOT)
