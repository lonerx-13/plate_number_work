import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse

# 添加当前目录到path，以便直接运行
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LPRNet, CHARS
from dataset import LPRDataset, collate_fn

def train_lpr(resume=False, weights=None):
    # 配置
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集
    train_dir = 'lpr_dataset/train'
    # train_dir = 'lpr_dataset/val'
    
    if not os.path.exists(train_dir):
        print(f"错误: 找不到数据集 {train_dir}")
        return

    train_dataset = LPRDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = LPRNet().to(DEVICE)
    criterion = nn.CTCLoss(blank=len(CHARS)-1, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    start_epoch = 0
    checkpoint_path = 'lpr_runs/lprnet_last.pth'
    
    if resume:
        if os.path.exists(checkpoint_path):
            print(f"正在从 {checkpoint_path} 恢复训练...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                print(f"成功恢复! 将从第 {start_epoch+1} 轮继续训练")
            except Exception as e:
                print(f"恢复失败: {e}, 将重新开始")
        else:
            print(f"未找到检查点 {checkpoint_path}, 将重新开始")
    
    model_path = weights if weights else None
    print(f"使用初始权重: {model_path}" if model_path else "使用随机初始化权重")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)) if model_path else None

    print(f"开始训练LPRNet, 设备: {DEVICE}")
    
    # 创建保存目录
    os.makedirs('lpr_runs', exist_ok=True)
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (imgs, labels, lengths) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs) # [B, W, C]
            
            # CTC Loss expects [T, B, C]
            outputs = outputs.permute(1, 0, 2)
            
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=outputs.size(0), dtype=torch.long)
            
            loss = criterion(outputs.log_softmax(2), labels, input_lengths, lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")
                
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # 保存完整检查点 (用于恢复)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型 (仅权重，用于推理)
        # 这里简单地每次都保存为best，实际应该比较loss或acc
        torch.save(model.state_dict(), 'lpr_runs/lprnet_best.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='指定初始权重路径 (例如 lpr_runs/lprnet_best.pth)')
    parser.add_argument('--resume', action='store_true', help='从上次中断的检查点恢复训练')
    args = parser.parse_args()
    train_lpr(resume=args.resume, weights=args.weights)
