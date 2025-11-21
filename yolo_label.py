import os
import cv2
from tqdm import tqdm

# CCPD数据集的字符映射
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def parse_filename(filename):
    # ... existing code ...
    base_name = filename.split('.')[0]
    parts = base_name.split('-')
    bbox_str = parts[2]
    coords = bbox_str.split('_')
    x1, y1 = map(int, coords[0].split('&'))
    x2, y2 = map(int, coords[1].split('&'))
    bbox = [(x1, y1), (x2, y2)]
    lp_number_indices_str = parts[4]
    lp_number_indices = [int(i) for i in lp_number_indices_str.split('_')]
    lp_number = ""
    lp_number += provinces[lp_number_indices[0]]
    lp_number += alphabets[lp_number_indices[1]]
    for i in range(2, 7):
        lp_number += ads[lp_number_indices[i]]
    return bbox, lp_number

def convert_to_yolo_format(bbox, img_width, img_height):
    """
    将边界框坐标转换为YOLO格式。
    """
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    # 计算中心点、宽度和高度
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    
    # 归一化
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    # 车牌只有一类，所以类别索引是0
    class_index = 0
    
    return f"{class_index} {x_center_norm} {y_center_norm} {width_norm} {height_norm}"

def process_ccpd_dataset(image_dir, label_dir):
    """
    处理CCPD数据集，生成YOLO格式的标签文件。
    """
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    for filename in tqdm(image_files, desc="Processing images"):
        try:
            # 解析文件名
            bbox, _ = parse_filename(filename)
            
            # 读取图片获取尺寸
            image_path = os.path.join(image_dir, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {filename}. Skipping.")
                continue
            img_height, img_width, _ = img.shape
            
            # 转换为YOLO格式
            yolo_label = convert_to_yolo_format(bbox, img_width, img_height)
            
            # 保存标签文件
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write(yolo_label)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义数据集目录和标签输出目录
    # 修正：使用 ccpd_base 作为图片源目录
    image_source_dir = os.path.join('CCPD2019', 'ccpd_base')
    labels_output_dir = os.path.join('CCPD2019', 'ccpd_base_labels')
    
    # 确保图片目录存在
    if not os.path.exists(image_source_dir):
        print(f"Error: Image directory not found at '{image_source_dir}'")
    else:
        # 执行处理
        process_ccpd_dataset(image_source_dir, labels_output_dir)
        print(f"\nProcessing complete. YOLO labels are saved in '{labels_output_dir}'.")