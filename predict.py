from ultralytics import YOLO
import cv2
import os

# --- 配置 ---
# 模型路径 (指向你训练好的 best.pt 文件)
MODEL_PATH = os.path.join('runs', 'detect', 'train3', 'weights', 'best.pt')

# 要预测的图片路径 (请替换为你自己的图片路径)
SOURCE_IMAGE_PATH = os.path.join('CCPD2019', 'ccpd_rotate', '035-22_18-294&457_468&625-468&555_302&625_294&527_460&457-0_0_22_30_2_31_27-139-54.jpg')

# --- 加载模型 ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"请确保模型文件路径正确: {MODEL_PATH}")
    exit()

# --- 进行预测 ---
# model.predict 会返回一个结果列表，因为它可以处理多个源
results = model.predict(source=SOURCE_IMAGE_PATH)

# --- 处理结果 ---
# 加载原始图片用于绘制
image = cv2.imread(SOURCE_IMAGE_PATH)

# 结果列表只有一个元素，因为我们只预测了一张图片
result = results[0]

print(f"在图片 {os.path.basename(SOURCE_IMAGE_PATH)} 中检测到 {len(result.boxes)} 个目标。")

# 遍历每个检测到的边界框
for box in result.boxes:
    # 获取边界框坐标 (xyxy格式：左上角x, 左上角y, 右下角x, 右下角y)
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy

    # 获取置信度
    confidence = box.conf[0].cpu().numpy()

    # 获取类别ID
    class_id = int(box.cls[0].cpu().numpy())
    class_name = model.names[class_id]

    # 在图片上绘制边界框
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 准备要显示的标签文字
    label = f'{class_name} {confidence:.2f}'

    # 在边界框上方绘制标签背景
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
    # 写入标签文字
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# --- 显示和保存结果 ---
# 创建保存结果的目录
output_dir = 'prediction_results'
os.makedirs(output_dir, exist_ok=True)

# 保存带有边界框的图片
output_path = os.path.join(output_dir, os.path.basename(SOURCE_IMAGE_PATH))
cv2.imwrite(output_path, image)

print(f"预测结果已保存至: {output_path}")

# 如果在图形界面环境中，可以取消下面的注释来直接显示图片
# cv2.imshow('Prediction Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()