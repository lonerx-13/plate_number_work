import torch
import cv2
import numpy as np
import sys
import os

# 添加当前目录到path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LPRNet, CHARS

def decode(preds):
    # preds: [T, B, C] or [B, T, C]
    # greedy decode
    pred_labels = []
    for i in range(preds.size(0)):
        pred = preds[i] # [T, C]
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

def predict_lpr(model_path, img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LPRNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (94, 24))
    img = img.astype('float32') / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img) # [1, W, C]
        result = decode(output)
        return result[0]

if __name__ == "__main__":
    # Test
    print(predict_lpr('lpr_runs/lprnet_best.pth', 'test_plate.jpg'))
