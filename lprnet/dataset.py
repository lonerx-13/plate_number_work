import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import sys

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import CHARS

class LPRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.char_to_idx = {c: i for i, c in enumerate(CHARS)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.root_dir, filename)
        
        # 读取图片
        # cv2.imdecode for chinese path
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (94, 24))
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1) # HWC -> CHW
        
        # 解析标签 (文件名即标签: 皖A12345_xxxx.jpg)
        label_str = filename.split('_')[0]
        label = []
        for c in label_str:
            if c in self.char_to_idx:
                label.append(self.char_to_idx[c])
        
        return torch.tensor(img), torch.tensor(label, dtype=torch.long), len(label)

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths)
    return imgs, labels, lengths
