# 车牌识别系统完整训练与开发指南 (进阶版)

本指南涵盖了大作业的所有功能需求 (1-6)，包括：
1.  **车牌检测与定位** (YOLOv8-Pose)
2.  **透视变换与矫正** (基于关键点)
3.  **字符识别** (LPRNet)
4.  **属性识别** (颜色、字符数)
5.  **GUI界面设计** (Tkinter)
6.  **生成EXE可执行文件**

## 1. 环境准备

请确保安装了 `requirements.txt` 中的所有依赖，并额外安装 `pyinstaller` 用于打包：

```bash
pip install -r requirements.txt
pip install pyinstaller
```

## 2. 第一阶段：车牌检测与定位 (YOLOv8-Pose)

为了实现透视变换，我们需要检测车牌的四个角点，因此使用 **YOLOv8-Pose** (关键点检测) 模型，而不是普通的检测模型。

### 2.1 数据转换
运行以下命令将CCPD数据集转换为YOLO Pose格式：

```bash
python ccpd_to_yolo_pose.py
```
*注意*: 请修改脚本中的 `CCPD_ROOT` 为你的数据集路径。

### 2.2 训练Pose模型
```bash
python train_pose.py
```
训练完成后，最佳模型保存在 `ccpd_pose_runs/exp1/weights/best.pt`。

### 2.3 测试矫正效果
运行以下命令测试检测和透视变换矫正：
```bash
python predict_pose.py
```
它会输出矫正后的矩形车牌图片到 `output_rectified` 目录。

## 3. 第二阶段：字符识别 (LPRNet)

使用 **LPRNet** (License Plate Recognition Network) 进行端到端的字符识别。

### 3.1 准备OCR数据集
我们需要利用CCPD的标注信息，将车牌裁剪出来并矫正，作为LPRNet的训练数据。
运行：
```bash
python lprnet/ccpd_to_lpr.py
```
这会在 `lpr_dataset` 目录下生成训练集和验证集。

### 3.2 训练LPRNet
```bash
python lprnet/train.py
```
训练完成后，最佳模型保存在 `lpr_runs/lprnet_best.pth`。

### 3.3 测试识别
```bash
python lprnet/predict.py
```

## 4. 第三阶段：系统集成与GUI

`gui.py` 集成了上述两个模型，提供图形化界面。

### 4.1 运行GUI
确保你已经训练好了两个模型 (或者有预训练模型)，然后运行：
```bash
python gui.py
```

### 4.2 功能说明
*   **选择图片**: 加载本地图片。
*   **开始识别**:
    1.  调用 YOLO Pose 模型检测车牌及4个角点。
    2.  根据角点进行透视变换，将倾斜的车牌矫正为正面矩形 (240x80)。
    3.  调用 LPRNet 模型识别车牌字符。
    4.  通过HSV颜色空间识别车牌颜色。
    5.  统计字符数量。
    6.  在界面上显示原图、矫正图和识别结果。

## 5. 第四阶段：生成EXE文件

使用 PyInstaller 将 Python 脚本打包为独立的可执行文件。

### 5.1 打包命令
在 `license_plate_training` 目录下运行：

```bash
pyinstaller --noconfirm --onefile --windowed --name "LicensePlateRecognizer" \
    --add-data "ccpd_pose_runs/exp1/weights/best.pt:." \
    --add-data "lpr_runs/lprnet_best.pth:." \
    gui.py
```

*注意*:
*   Windows下分隔符为 `;` (即 `--add-data "src;dest"`)，Linux/Mac下为 `:`。
*   请确保模型路径正确。如果还没训练，可以使用占位文件测试打包。
*   打包完成后，EXE文件位于 `dist/` 目录下。

## 6. 常见问题

*   **模型加载失败**: 请检查路径是否正确，或者是否已经完成了训练。
*   **中文乱码**: 代码中已处理中文路径问题，但如果遇到显示乱码，请检查系统字体。
*   **显存不足**: 在 `train_pose.py` 和 `lprnet/train.py` 中调小 `BATCH_SIZE`。

祝大作业取得高分！
