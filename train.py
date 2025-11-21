import os
from ultralytics import YOLO

def main():
    # 在导入任何可能使用OpenMP的库（如torch, numpy）之前设置环境变量
    # 这是解决 OMP: Error #15 的关键
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 加载一个预训练模型 (推荐从yolov8n.pt开始)
    model = YOLO('runs/detect/train3/weights/last.pt')

    # 使用您的数据集配置文件开始训练
    # 请确保 ccpd_dataset.yaml 文件在当前目录下
    results = model.train(
        data='ccpd_dataset.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16,
        device = 0, 
        resume = True,
    )

if __name__ == '__main__':
    main()