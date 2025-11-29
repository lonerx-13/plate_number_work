# 车牌识别系统

基于深度学习的中国车牌识别系统，支持蓝牌、黄牌、绿牌（新能源）识别。

## 功能特性

- 🚗 **车牌检测**：基于 YOLOv8-Pose 的车牌定位与四点矫正
- 🎨 **颜色识别**：ColorNet 识别蓝牌/黄牌/绿牌
- 🔤 **车牌识别**：支持多种识别方法
  - **LPRNet**：端到端深度学习识别（推荐，适合复杂场景）
  - **HOG 模板匹配**：传统特征匹配（适合高清图片）
- 🖼️ **GUI 界面**：直观的图形操作界面

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 模型文件

确保以下模型文件存在：
- `yolov8n-pose.pt` - 车牌检测模型
- `lprnet/best.pth` - LPRNet 车牌识别模型
- `colornet/best.pth` - 颜色识别模型
- `character/` - 字符模板库（HOG 匹配使用）

## 运行

```bash
python gui.py
```

## 使用说明

1. 点击 **选择图片** 加载车牌图片
2. 选择 **识别方法**：
   - `lprnet`：深度学习端到端识别（推荐）
   - `hog_euclidean`：HOG 特征 + 欧氏距离匹配
   - `hog_cosine`：HOG 特征 + 余弦相似度匹配
3. 点击 **开始识别** 查看结果

> 💡 **提示**：LPRNet 适合监控摄像头、模糊、倾斜等复杂场景；HOG 方法适合手机拍摄的高清正面车牌。

## 项目结构

```
plate_number_work/
├── gui.py                  # GUI 界面主程序
├── requirements.txt        # 依赖列表
├── yolov8n-pose.pt         # YOLO 车牌检测模型
│
├── lprnet/                 # LPRNet 端到端识别模块
│   ├── model.py            # 模型定义
│   └── best.pth            # 预训练权重
│
├── colornet/               # 车牌颜色识别模块
│   ├── model.py            # 模型定义
│   └── best.pth            # 预训练权重
│
├── template_match/         # HOG 模板匹配识别模块
│   ├── __init__.py         # 模块导出
│   ├── recognizer.py       # 识别器封装
│   ├── segment.py          # 基础分割方法
│   ├── segment_v3.py       # V3 分割（连通区域分析）
│   ├── matcher.py          # 模板匹配器
│   └── deep_feature.py     # HOG 特征提取与匹配
│
├── character/              # 字符模板库
│   ├── chinese/            # 汉字模板（省份简称）
│   └── alphanumeric/       # 字母数字模板
│
├── utils/                  # 工具函数
├── Pose/                   # 姿态估计相关
│
├── analyze_hog.py          # HOG 匹配分析脚本
├── test_ccpd.py            # CCPD 数据集测试脚本
├── expand_templates.py     # 模板库扩充脚本
├── train_pose.py           # 姿态模型训练脚本
└── ccpd_to_yolo_pose.py    # CCPD 数据集转换脚本
```

## 识别方法对比

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| LPRNet | 端到端深度学习 | 准确率高、鲁棒性强、速度快 | 需要 GPU 加速 | 复杂场景、实时识别 |
| HOG 模板匹配 | 特征提取 + 模板比对 | 可解释性强、无需训练 | 对图片质量要求高 | 高清图片、离线识别 |

## 字符分割（V3）

HOG 模板匹配使用 V3 分割算法：

1. **Canny 边缘检测**：获取字符边界
2. **连通区域分析**：提取候选字符区域
3. **智能合并**：处理汉字被分割成上下两部分的情况
4. **智能拆分**：处理两个靠近的字符被识别为一个的情况
5. **上半部分投影**：避免铆钉干扰

## 技术栈

- **深度学习框架**：PyTorch
- **目标检测**：YOLOv8 (Ultralytics)
- **图像处理**：OpenCV, scikit-image
- **GUI**：Tkinter
- **特征提取**：HOG (skimage)

## License

MIT License
