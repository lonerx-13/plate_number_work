# 车牌识别系统

## 安装

在终端中执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 运行

运行 GUI 界面：
```bash
python gui.py
```

## 字符分割模块

本项目提供了多种字符分割算法，位于 `template_match/` 目录下：

### 分割方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| `V2_PROJECTION` | 垂直投影法 | 标准车牌，字符清晰 |
| `V2_HYBRID` | 混合策略（投影+固定宽度） | 通用场景 |
| `V3_CONNECTED` | 连通区域分析法（Canny边缘检测） | 复杂背景，汉字分割 |
| `V3_FALLBACK` | V3 带回退机制 | **推荐使用** |
| `AUTO` | 自动选择最佳方法 | 不确定场景 |

### V3 分割特点

V3 版本（`segment_v3.py`）采用连通区域分析方法：

1. **Canny 边缘检测**：获取字符边界
2. **连通区域分析**：提取候选字符区域
3. **智能合并**：处理汉字被分割成上下两部分的情况
4. **智能拆分**：处理两个靠近的字符被识别为一个的情况
5. **上半部分投影**：避免铆钉干扰（使用上 60% 区域计算投影）

### 使用示例

```python
from template_match import TemplateRecognizer, SegmentMethod

# 创建识别器，使用 V3 带回退机制的分割
recognizer = TemplateRecognizer(segment_method=SegmentMethod.V3_FALLBACK)

# 也可以动态切换分割方法
recognizer.set_segment_method(SegmentMethod.V3_CONNECTED)

# 识别车牌
result = recognizer.recognize_with_details(plate_img)
print(f"车牌号: {result['plate_number']}")
```

### 直接使用分割函数

```python
from template_match import segment_characters_v3, visualize_segmentation_v3

# 使用 V3 分割（返回字符列表和调试信息）
chars, debug_info = segment_characters_v3(plate_img, expected_chars=7, debug=True)

# 可视化分割结果
visualize_segmentation_v3(plate_img, chars, debug_info)
```

## 项目结构

```
plate_number_work/
├── gui.py                 # GUI 界面
├── template_match/        # 模板匹配识别模块
│   ├── segment.py         # V1 分割（基础版）
│   ├── segment_v2.py      # V2 分割（投影法）
│   ├── segment_v3.py      # V3 分割（连通区域分析）★
│   ├── matcher.py         # 模板匹配器
│   ├── recognizer.py      # 完整识别器
│   └── deep_feature.py    # HOG 特征匹配
├── lprnet/                # LPRNet 端到端识别
├── colornet/              # 车牌颜色识别
└── character/             # 字符模板库
```
