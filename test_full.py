"""
完整车牌识别测试脚本
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from template_match.segment_lpr import split_char  # 使用 LPR 项目的分割算法
from cnn_classifier import CNNCharRecognizer

def test_plate(image_path):
    # 加载模型
    yolo_path = 'ccpd_pose_runs/exp1/weights/best.pt'
    model = YOLO(yolo_path)
    recognizer = CNNCharRecognizer()

    # 读取测试图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    print(f'图片尺寸: {img.shape}')

    # YOLO 检测
    results = model(img, verbose=False)
    result = results[0]

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        kpts = result.keypoints.xy.cpu().numpy()[0].astype(np.float32)
        print(f'关键点: {kpts}')
        
        # 透视变换
        dst_pts = np.array([[0, 0], [240, 0], [240, 80], [0, 80]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(kpts, dst_pts)
        warped = cv2.warpPerspective(img, M, (240, 80))
        
        cv2.imwrite('warped_plate.png', warped)
        print('矫正后的车牌已保存到 warped_plate.png')
        
        # 使用 LPR 分割算法
        chars = split_char(warped)
        print(f'分割出 {len(chars)} 个字符')
        
        # 保存分割后的字符
        for i, char_img in enumerate(chars):
            cv2.imwrite(f'char_{i}.png', char_img)
            h, w = char_img.shape[:2]
            mean_val = np.mean(char_img)
            print(f'  字符{i}: {w}x{h}, mean={mean_val:.1f}')
        
        if len(chars) < 2:
            print("分割失败")
            return
        
        # CNN 识别
        result = recognizer.recognize_with_details(chars)
        print(f'\n识别结果: {result["plate_number"]}')
        print(f'置信度: {result["confidence"]:.1%}')
        for i, (char, conf) in enumerate(result['char_details']):
            print(f'  {i}: {char} ({conf:.1%})')
    else:
        print('未检测到车牌')

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_plate(sys.argv[1])
    else:
        test_plate('test_plate.png')
