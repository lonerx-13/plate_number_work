import os
import random
import shutil
from tqdm import tqdm

def split_dataset(image_source_dir, label_source_dir, base_output_dir, train_ratio=0.8):
    """
    将数据集划分为训练集和验证集。
    """
    if not os.path.exists(image_source_dir):
        print(f"Error: Image source directory '{image_source_dir}' not found.")
        return

    if not os.path.exists(label_source_dir):
        print(f"Error: Label source directory '{label_source_dir}' not found.")
        return

    # 创建目标目录
    train_img_dir = os.path.join(base_output_dir, 'images', 'train')
    val_img_dir = os.path.join(base_output_dir, 'images', 'val')
    train_label_dir = os.path.join(base_output_dir, 'labels', 'train')
    val_label_dir = os.path.join(base_output_dir, 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # 获取所有图片文件名
    image_files = [f for f in os.listdir(image_source_dir) if f.endswith('.jpg')]
    random.shuffle(image_files)

    # 计算划分点
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 移动文件
    print("Copying training files...")
    for filename in tqdm(train_files, desc="Train Set"):
        shutil.copy(os.path.join(image_source_dir, filename), os.path.join(train_img_dir, filename))
        label_filename = os.path.splitext(filename)[0] + '.txt'
        shutil.copy(os.path.join(label_source_dir, label_filename), os.path.join(train_label_dir, label_filename))

    print("Copying validation files...")
    for filename in tqdm(val_files, desc="Validation Set"):
        shutil.copy(os.path.join(image_source_dir, filename), os.path.join(val_img_dir, filename))
        label_filename = os.path.splitext(filename)[0] + '.txt'
        shutil.copy(os.path.join(label_source_dir, label_filename), os.path.join(val_label_dir, label_filename))

    print("Dataset splitting complete.")

def create_yolo_config(output_dir, dataset_path):
    """
    创建YOLOv8的数据集配置文件 (dataset.yaml)。
    """
    config_path = os.path.join(output_dir, 'ccpd_dataset.yaml')
    dataset_path = os.path.abspath(dataset_path)
    
    config_content = f"""\
    train: {os.path.join(dataset_path, 'images', 'train').replace('\\', '/')}
    val: {os.path.join(dataset_path, 'images', 'val').replace('\\', '/')}

    # Number of classes
    nc: 1

    # Class names
    names: ['license_plate']
    """

    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"YOLO config file created at: {config_path}")

if __name__ == "__main__":
    image_source_dir = os.path.join('CCPD2019', 'ccpd_base')
    label_source_dir = os.path.join('CCPD2019', 'ccpd_base_labels')
    dataset_output_dir = 'CCPD2019' # 数据集将被整理到这个文件夹下

    split_dataset(image_source_dir, label_source_dir, dataset_output_dir)
    create_yolo_config(os.getcwd(), dataset_output_dir) # 在当前目录创建yaml