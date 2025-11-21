import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- 配置 ---
# 1. 模型路径 (使用您训练好的最佳模型)
MODEL_PATH = 'runs/detect/train3/weights/best.pt'

# 2. 源图片目录 (您想要从中检测和裁剪车牌的图片)
SOURCE_DIR = 'CCPD2019/ccpd_rotate'

# 3. 保存裁剪后车牌的目录
OUTPUT_DIR = 'ccpd_rotate/cropped_plates'

# 4. 置信度阈值 (只保存检测结果大于此值的车牌)
CONF_THRESHOLD = 0.7

# --- 主程序 ---
def crop_license_plates():
    """加载YOLO模型，检测并裁剪图片中的车牌。"""
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    # 加载YOLOv8模型
    try:
        model = YOLO(MODEL_PATH)
        print(f"成功加载模型: {MODEL_PATH}")
    except Exception as e:
        print(f"错误: 无法加载模型，请检查路径是否正确: {MODEL_PATH}")
        print(e)
        return

    # 获取所有图片文件
    try:
        image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"错误: 在目录 {SOURCE_DIR} 中未找到图片文件。")
            return
        print(f"在 {SOURCE_DIR} 中找到 {len(image_files)} 张图片，开始处理...")
    except FileNotFoundError:
        print(f"错误: 源目录不存在: {SOURCE_DIR}")
        return

    # 遍历所有图片并进行处理
    for filename in tqdm(image_files, desc="裁剪车牌中"):
        image_path = os.path.join(SOURCE_DIR, filename)
        
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图片 {image_path}")
            continue

        # 使用模型进行预测
        results = model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)

        # 遍历检测结果
        for i, result in enumerate(results):
            # 获取边界框
            boxes = result.boxes.xyxy.cpu().numpy() # .xyxy格式为 (xmin, ymin, xmax, ymax)
            
            if len(boxes) == 0:
                continue

            # 裁剪并保存每个检测到的车牌
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # 裁剪车牌区域
                cropped_plate = img[y1:y2, x1:x2]
                
                # 构建保存路径
                base_filename = os.path.splitext(filename)[0]
                output_filename = f"{base_filename}_plate_{j}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # 保存裁剪后的图片
                cv2.imwrite(output_path, cropped_plate)

    print(f"\n处理完成！所有裁剪后的车牌已保存到 '{OUTPUT_DIR}' 目录中。")

if __name__ == '__main__':
    crop_license_plates()