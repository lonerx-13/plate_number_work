import os
from tqdm import tqdm

# --- 配置 --- #
# 请将此路径修改为您存放裁切后车牌图像的文件夹
# 根据您的描述，这里应该设置为 'ccpd_rotate'
CROPPED_PLATES_DIR = 'ccpd_rotate/cropped_plates' 
# 输出的标签文件名
LABEL_FILE = 'rec_labels.txt'
# --- 配置结束 --- #

# 根据您提供的更完整的CCPD数据集字符映射
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def parse_plate_number(filename):
    """从 ccpd_rotate 数据集的文件名中解析车牌号码"""
    try:
        # 移除文件名末尾的 '_plate_0.jpg'
        base_name = filename.replace('_plate_0.jpg', '')
        
        # 文件名格式: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15
        # 我们需要的部分是倒数第三段
        parts = base_name.split('-')
        plate_indices_str = parts[4] # 车牌号码在第5个字段
        plate_indices = [int(i) for i in plate_indices_str.split('_')]

        plate_number = ""
        # 第一个字符是省份
        plate_number += provinces[plate_indices[0]]
        # 第二个字符是字母
        plate_number += alphabets[plate_indices[1]]
        # 后面5个是字母或数字
        for i in range(2, 7):
            # 检查索引是否有效，防止'O'作为占位符导致的问题
            if plate_indices[i] < len(ads) and ads[plate_indices[i]] != 'O':
                plate_number += ads[plate_indices[i]]
        
        return plate_number
    except (IndexError, ValueError) as e:
        print(f"警告：无法解析文件名 '{filename}'。跳过此文件。错误: {e}")
        return None

def create_label_file():
    """创建标签文件"""
    if not os.path.isdir(CROPPED_PLATES_DIR):
        print(f"错误：文件夹 '{CROPPED_PLATES_DIR}' 不存在。请检查 CROPPED_PLATES_DIR 变量设置是否正确。")
        return

    image_files = [f for f in os.listdir(CROPPED_PLATES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"警告：在文件夹 '{CROPPED_PLATES_DIR}' 中没有找到任何图像文件。")
        return

    print(f"正在从 '{CROPPED_PLATES_DIR}' 文件夹中的 {len(image_files)} 个图像文件创建标签...")

    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        for filename in tqdm(image_files, desc="生成标签"):
            plate_number = parse_plate_number(filename)
            if plate_number:
                # 写入相对路径和标签，用制表符分隔
                relative_path = os.path.join(CROPPED_PLATES_DIR, filename).replace('\\', '/')
                f.write(f"{relative_path}\t{plate_number}\n")

    print(f"标签文件 '{LABEL_FILE}' 创建成功！")
    print(f"下一步，您可以使用这个文件来训练您的字符识别模型。")

if __name__ == '__main__':
    create_label_file()