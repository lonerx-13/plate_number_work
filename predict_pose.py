from ultralytics import YOLO
import cv2
import numpy as np
import os

def order_points(pts):
    # 确保点顺序为 TL, TR, BR, BL
    # 我们的模型训练目标就是 TL, TR, BR, BL，所以直接使用即可
    # 但为了保险，可以根据坐标排序
    # 这里假设模型输出顺序正确
    return pts

def four_point_transform(image, pts):
    # 目标尺寸 (宽, 高)
    # 中国车牌标准尺寸 440mm x 140mm -> 比例约 3.14
    # 设定输出图片大小为 240 x 80
    width = 240
    height = 80
    
    # 目标点: TL, TR, BR, BL
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")
    
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst)
    
    # 进行变换
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def predict_and_rectify(model_path, image_path, output_dir='output_rectified'):
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图片")
        return

    results = model(img)
    result = results[0]
    
    if result.keypoints is None:
        print("未检测到关键点")
        return
        
    # 获取关键点
    # shape: (N, 4, 2) or (N, 4, 3)
    keypoints = result.keypoints.xy.cpu().numpy()
    
    for i, kpts in enumerate(keypoints):
        # kpts shape: (4, 2) -> TL, TR, BR, BL
        # 转换为float32
        pts = np.array(kpts, dtype="float32")
        
        # 透视变换
        warped = four_point_transform(img, pts)
        
        save_path = os.path.join(output_dir, f"rectified_{i}.jpg")
        cv2.imwrite(save_path, warped)
        print(f"已保存矫正后的车牌: {save_path}")

if __name__ == "__main__":
    MODEL_PATH = 'ccpd_pose_runs/exp1/weights/best.pt'
    TEST_IMG = 'test.jpg'
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = 'yolov8n-pose.pt' # Fallback
    
    if os.path.exists(TEST_IMG):
        predict_and_rectify(MODEL_PATH, TEST_IMG)
