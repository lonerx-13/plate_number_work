import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
from lprnet.model import LPRNet, CHARS
from colornet.model import ColorNet

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
        self.root.geometry("1000x700")
        
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
                
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")

    def setup_ui(self):
        # 顶部按钮
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="选择图片", command=self.load_image, width=15, height=2).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="开始识别", command=self.process_image, width=15, height=2).pack(side=tk.LEFT, padx=10)
        
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
        
        # 结果文本
        res_frame = tk.Frame(self.root)
        res_frame.pack(pady=20, fill=tk.X, padx=20)
        
        tk.Label(res_frame, text="识别结果:", font=("Arial", 14)).pack(anchor=tk.W)
        self.txt_result = tk.Text(res_frame, height=5, font=("Arial", 12))
        self.txt_result.pack(fill=tk.X)
        
        self.current_image_path = None
        self.current_image = None

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
            
            # 4. LPRNet 识别
            plate_text = "模型未加载"
            if self.lpr_model:
                lpr_input = cv2.resize(warped, (94, 24))
                lpr_input = lpr_input.astype('float32') / 255.0
                lpr_input = lpr_input.transpose(2, 0, 1)
                lpr_input = torch.tensor(lpr_input).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.lpr_model(lpr_input)
                    plate_text = decode(output)[0]
            
            # 输出结果
            info = f"车牌 {i+1}:\n"
            info += f"  号码: {plate_text}\n"
            info += f"  颜色: {color}\n"
            info += f"  字符数: {len(plate_text)}\n"
            self.txt_result.insert(tk.END, info)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
