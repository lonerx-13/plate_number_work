import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
from lprnet.model import LPRNet, CHARS
from colornet.model import ColorNet
from template_match.recognizer import TemplateRecognizer, SegmentMethod
from template_match.segment_v3 import segment_with_fallback

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 导入辅助函数
def four_point_transform(image, pts):
    width = 240
    height = 80
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def decode(preds):
    pred_labels = []
    for i in range(preds.size(0)):
        pred = preds[i]
        pred_indices = torch.argmax(pred, dim=1)
        char_list = []
        prev_idx = -1
        for idx in pred_indices:
            idx = idx.item()
            if idx != prev_idx and idx != len(CHARS)-1:
                char_list.append(CHARS[idx])
            prev_idx = idx
        pred_labels.append("".join(char_list))
    return pred_labels

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别系统 (大作业)")
        self.root.geometry("1200x750")
        
        # 模型加载
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
        
        # UI布局
        self.setup_ui()
        
    def load_models(self):
        try:
            # YOLO Pose
            local_yolo = 'ccpd_pose_runs/exp1/weights/best.pt'
            if os.path.exists(local_yolo):
                self.yolo_path = local_yolo
            else:
                self.yolo_path = resource_path('best_pose.pt')
                
            if not os.path.exists(self.yolo_path):
                self.yolo_path = 'yolov8n-pose.pt'
                print("使用预训练YOLO模型")
            
            self.yolo_model = YOLO(self.yolo_path)
            
            # LPRNet
            local_lpr = 'lpr_runs/lprnet_best.pth'
            if os.path.exists(local_lpr):
                self.lpr_path = local_lpr
            else:
                self.lpr_path = resource_path('lprnet_best.pth')
                
            self.lpr_model = LPRNet().to(self.device)
            if os.path.exists(self.lpr_path):
                self.lpr_model.load_state_dict(torch.load(self.lpr_path, map_location=self.device))
                self.lpr_model.eval()
            else:
                print("警告: 未找到LPRNet模型，识别功能将不可用")
                self.lpr_model = None
                
            # ColorNet
            local_color = 'color_runs/color_best.pth'
            if os.path.exists(local_color):
                self.color_path = local_color
            else:
                self.color_path = resource_path('color_best.pth')
            
            self.color_model = ColorNet().to(self.device)
            if os.path.exists(self.color_path):
                self.color_model.load_state_dict(torch.load(self.color_path, map_location=self.device))
                self.color_model.eval()
            else:
                print("警告: 未找到ColorNet模型，颜色识别将使用HSV兜底")
                self.color_model = None
            
            # HOG 特征识别器（延迟加载）
            template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character")
            self.template_dir = template_dir
            self.template_recognizer_deep = None  # 延迟加载 HOG 特征版本
            if os.path.exists(template_dir):
                print("字符模板目录已找到，HOG 识别器将延迟加载")
            else:
                print("警告: 未找到字符模板目录，HOG 识别不可用")
                
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")

    def setup_ui(self):
        # 顶部按钮
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="选择图片", command=self.load_image, width=15, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="开始识别", command=self.process_image, width=15, height=2).pack(side=tk.LEFT, padx=10)
        
        # 识别方法选择
        method_frame = tk.Frame(btn_frame)
        method_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(method_frame, text="识别方法:").pack(side=tk.LEFT)
        self.recognition_method = tk.StringVar(value="lprnet")
        method_combo = ttk.Combobox(method_frame, textvariable=self.recognition_method, 
                                     values=["lprnet", "hog_euclidean", "hog_cosine"],
                                     state="readonly", width=18)
        method_combo.pack(side=tk.LEFT, padx=5)
        
        # 识别方法提示按钮
        tip_btn = tk.Button(method_frame, text="💡说明", font=("微软雅黑", 9),
                           command=self.show_method_help, bg="#4CAF50", fg="white",
                           activebackground="#45a049", cursor="hand2")
        tip_btn.pack(side=tk.LEFT, padx=5)
        
        # 分割方法固定使用 V3
        self.segmentation_method = tk.StringVar(value="v3_fallback")
        
        # 图片显示区域
        img_frame = tk.Frame(self.root)
        img_frame.pack(expand=True, fill=tk.BOTH, padx=20)
        
        # 原图
        self.lbl_orig = tk.Label(img_frame, text="原图")
        self.lbl_orig.grid(row=0, column=0, padx=10)
        self.canvas_orig = tk.Canvas(img_frame, width=400, height=300, bg='gray')
        self.canvas_orig.grid(row=1, column=0, padx=10)
        
        # 结果图 (裁剪+矫正)
        self.lbl_crop = tk.Label(img_frame, text="矫正后车牌")
        self.lbl_crop.grid(row=0, column=1, padx=10)
        self.canvas_crop = tk.Canvas(img_frame, width=240, height=80, bg='gray')
        self.canvas_crop.grid(row=1, column=1, padx=10)
        
        # 分割预览
        self.lbl_seg = tk.Label(img_frame, text="字符分割预览")
        self.lbl_seg.grid(row=0, column=2, padx=10)
        self.canvas_seg = tk.Canvas(img_frame, width=280, height=80, bg='white')
        self.canvas_seg.grid(row=1, column=2, padx=10)
        
        # 结果文本
        res_frame = tk.Frame(self.root)
        res_frame.pack(pady=20, fill=tk.X, padx=20)
        
        tk.Label(res_frame, text="识别结果:", font=("Arial", 14)).pack(anchor=tk.W)
        self.txt_result = tk.Text(res_frame, height=5, font=("Arial", 12))
        self.txt_result.pack(fill=tk.X)
        
        self.current_image_path = None
        self.current_image = None
        self.tk_seg = None  # 保持分割预览的引用
    
    def show_segmentation_preview(self, char_images):
        """显示字符分割预览"""
        if not char_images:
            return
        
        # 创建拼接图像
        target_h = 60  # 目标高度
        total_w = 0
        resized_chars = []
        
        for img in char_images:
            if img is None or img.size == 0:
                continue
            
            # 确保是灰度图
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            h, w = img.shape[:2]
            # 保持宽高比缩放
            scale = target_h / h
            new_w = max(int(w * scale), 10)
            resized = cv2.resize(img, (new_w, target_h))
            resized_chars.append(resized)
            total_w += new_w + 2  # +2 for border
        
        if not resized_chars:
            return
        
        # 确保 total_w 有效
        if total_w <= 0:
            return
        
        # 创建预览图（灰度图，白色背景）
        preview = np.ones((target_h, total_w), dtype=np.uint8) * 255
        x = 0
        for i, char_img in enumerate(resized_chars):
            h, w = char_img.shape[:2]
            # 边界检查
            if x + w > total_w:
                w = total_w - x
            if w > 0 and h > 0:
                preview[:h, x:x+w] = char_img[:h, :w]
            x += w + 2
        
        # 转换为 PIL 并显示
        preview_pil = Image.fromarray(preview)
        preview_pil = preview_pil.resize((280, 60))
        self.tk_seg = ImageTk.PhotoImage(preview_pil)
        self.canvas_seg.delete("all")
        self.canvas_seg.create_image(140, 40, image=self.tk_seg)

    def show_method_help(self):
        """显示识别方法帮助窗口"""
        help_window = tk.Toplevel(self.root)
        help_window.title("识别方法说明")
        help_window.geometry("450x320")
        help_window.resizable(False, False)
        
        # 居中显示
        help_window.transient(self.root)
        help_window.grab_set()
        
        help_text = """
【识别方法选择指南】
🔹 LPRNet（推荐）
   • 基于深度学习的端到端识别
   • 适合复杂场景：监控摄像头、模糊、倾斜、光照不均
   • 准确率高，鲁棒性强
   • 速度快，无需字符分割

🔹 HOG + 欧氏距离 / HOG + 余弦相似度
   • 基于 HOG 特征的模板匹配方法
   • 适合高清图片：手机拍摄的正面清晰车牌
   • 需要先进行字符分割
   • 对图片质量要求较高

📌 说明：
   • 一般场景优先使用 LPRNet
   • HOG 方法作为实验对照选项(HOG+余弦相似度通常效果更好)
        """
        
        text_label = tk.Label(help_window, text=help_text, justify=tk.LEFT, 
                             font=("微软雅黑", 10), padx=20, pady=10)
        text_label.pack(expand=True, fill=tk.BOTH)
        
        close_btn = tk.Button(help_window, text="知道了", command=help_window.destroy,
                             width=10, height=1)
        close_btn.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.current_image_path = path
            img = Image.open(path)
            img.thumbnail((400, 300))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas_orig.create_image(200, 150, image=self.tk_img)
            self.txt_result.delete(1.0, tk.END)

    def process_image(self):
        if not self.current_image_path:
            return
            
        # 1. YOLO 检测
        img_cv = cv2.imdecode(np.fromfile(self.current_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        results = self.yolo_model(img_cv)
        result = results[0]

        keypoints = result.keypoints.xy.cpu().numpy()
        if len(keypoints) == 0:
            self.txt_result.insert(tk.END, "未检测到车牌\n")
            return
        
        self.txt_result.delete(1.0, tk.END)
        
        for i, kpts in enumerate(keypoints):
            # 2. 矫正
            pts = np.array(kpts, dtype="float32")
            warped = four_point_transform(img_cv, pts)
            
            # 显示矫正图
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(warped_rgb)
            self.tk_crop = ImageTk.PhotoImage(im_pil)
            self.canvas_crop.create_image(120, 40, image=self.tk_crop)
            
            # 3. 颜色识别 (ColorNet)
            color = "未知"
            if self.color_model:
                c_input = cv2.resize(warped, (94, 24))
                c_input = c_input.astype('float32') / 255.0
                c_input = c_input.transpose(2, 0, 1)
                c_input = torch.tensor(c_input).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    c_out = self.color_model(c_input)
                    _, c_pred = c_out.max(1)
                    color_classes = ['蓝色', '黄色', '绿色']
                    color = color_classes[c_pred.item()]
            else:
                # HSV 兜底
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                mean_h = np.mean(hsv[:, :, 0])
                if 100 < mean_h < 124: color = "蓝色"
                elif 35 < mean_h < 77: color = "绿色"
                elif 11 < mean_h < 34: color = "黄色"
            
            # 4. 车牌号识别 (根据选择的方法)
            plate_text = "模型未加载"
            confidence = 0.0
            method = self.recognition_method.get()
            
            if method == "lprnet":
                # LPRNet 端到端识别（不需要字符分割）
                if self.lpr_model:
                    lpr_input = cv2.resize(warped, (94, 24))
                    lpr_input = lpr_input.astype('float32') / 255.0
                    lpr_input = lpr_input.transpose(2, 0, 1)
                    lpr_input = torch.tensor(lpr_input).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.lpr_model(lpr_input)
                        plate_text = decode(output)[0]
                        confidence = 1.0  # LPRNet 不直接输出置信度
                    
                    # LPRNet 是端到端识别，不显示分割预览
                    self.canvas_seg.delete("all")
                    self.canvas_seg.create_text(140, 40, text="LPRNet 端到端识别\n无需字符分割", 
                                               font=("微软雅黑", 9), fill="gray")
            
            elif method.startswith("hog_"):
                # HOG 特征匹配
                if hasattr(self, 'template_dir') and os.path.exists(self.template_dir):
                    # 延迟加载 HOG 特征识别器
                    if self.template_recognizer_deep is None:
                        self.txt_result.insert(tk.END, "首次加载 HOG 特征识别器，请稍候...\n")
                        self.root.update()
                        self.template_recognizer_deep = TemplateRecognizer(self.template_dir, use_deep_feature=True)
                    
                    # 设置匹配方法
                    if "euclidean" in method:
                        self.template_recognizer_deep.set_match_method("euclidean")
                    else:
                        self.template_recognizer_deep.set_match_method("cosine")
                    
                    # 固定使用 V3 分割方法
                    self.template_recognizer_deep.set_segment_method(SegmentMethod.V3_FALLBACK)
                    
                    # 根据颜色确定车牌类型
                    if color == "绿色":
                        plate_type = "green"
                    elif color == "黄色":
                        plate_type = "220"
                    else:
                        plate_type = "140"
                    
                    result = self.template_recognizer_deep.recognize_with_details(
                        warped, plate_type=plate_type
                    )
                    plate_text = result['plate_number']
                    confidence = result['confidence']
                    
                    # 显示分割预览
                    if 'char_images' in result and result['char_images']:
                        self.show_segmentation_preview(result['char_images'])
                else:
                    plate_text = "HOG 识别器不可用（模板目录不存在）"
            
            # 输出结果
            info = f"车牌 {i+1}:\n"
            info += f"  号码: {plate_text}\n"
            info += f"  颜色: {color}\n"
            info += f"  方法: {method}\n"
            if method.startswith("hog_"):
                info += f"  置信度: {confidence:.1%}\n"
            info += f"  字符数: {len(plate_text)}\n"
            self.txt_result.insert(tk.END, info)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
