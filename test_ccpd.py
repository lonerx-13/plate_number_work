"""
CCPD 数据集测试脚本
支持三种识别方法的对比测试：lprnet, hog_euclidean, hog_cosine

CCPD 文件名格式:
025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
字段说明:
1. 面积比例
2. 倾斜角度
3. 边界框坐标 (左上&右下)
4. 四个顶点坐标 (右下_左下_左上_右上)
5. 车牌号索引 (省份_字母_5个字符)
6. 亮度
7. 模糊度

使用方法:
1. 下载 CCPD 数据集并解压
2. 修改 CCPD_ROOT 为你的数据集路径
3. 运行: python test_ccpd.py
"""

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse
import time

# CCPD 字符映射表
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", 
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", 
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", 
             "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 
             'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', 
       '4', '5', '6', '7', '8', '9', 'O']


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名，提取车牌信息
    
    Returns:
        dict: {
            'plate_number': str,  # 车牌号
            'vertices': np.array,  # 四个顶点坐标 (4, 2)
            'bbox': tuple,  # (x1, y1, x2, y2)
        } 或 None（解析失败）
    """
    try:
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('-')
        if len(parts) < 5:
            return None
        
        # 解析车牌号 (第5个字段)
        plate_indices = parts[4].split('_')
        if len(plate_indices) < 7:
            return None
            
        plate_str = PROVINCES[int(plate_indices[0])]
        plate_str += ALPHABETS[int(plate_indices[1])]
        for idx in plate_indices[2:]:
            plate_str += ADS[int(idx)]
        
        # 解析边界框 (第3个字段)
        bbox_str = parts[2]
        bbox_parts = bbox_str.split('_')
        x1, y1 = map(int, bbox_parts[0].split('&'))
        x2, y2 = map(int, bbox_parts[1].split('&'))
        
        # 解析四个顶点 (第4个字段)
        # 顺序: 右下_左下_左上_右上
        vertices_str = parts[3]
        v_coords = vertices_str.split('_')
        vertices = []
        for v in v_coords:
            vx, vy = v.split('&')
            vertices.append([int(vx), int(vy)])
        
        # 转换顺序: 左上, 右上, 右下, 左下 (用于透视变换)
        # CCPD: BR(0), BL(1), TL(2), TR(3) -> TL, TR, BR, BL
        vertices = np.array([vertices[2], vertices[3], vertices[0], vertices[1]], dtype="float32")
        
        return {
            'plate_number': plate_str,
            'vertices': vertices,
            'bbox': (x1, y1, x2, y2),
        }
    except Exception as e:
        return None


def four_point_transform(image, pts, width=240, height=80):
    """透视变换矫正车牌"""
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


class CCPDTester:
    """CCPD 数据集测试器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpr_model = None
        self.template_recognizer = None
        self.load_models()
    
    def load_models(self):
        """加载所有模型"""
        # 加载 LPRNet
        try:
            from lprnet.model import LPRNet, CHARS
            self.CHARS = CHARS
            lpr_path = 'lpr_runs/lprnet_best.pth'
            if os.path.exists(lpr_path):
                self.lpr_model = LPRNet().to(self.device)
                self.lpr_model.load_state_dict(torch.load(lpr_path, map_location=self.device))
                self.lpr_model.eval()
                print(f"✓ LPRNet 模型已加载: {lpr_path}")
            else:
                print(f"✗ LPRNet 模型不存在: {lpr_path}")
        except Exception as e:
            print(f"✗ LPRNet 加载失败: {e}")
        
        # 加载 HOG 模板识别器
        try:
            from template_match.recognizer import TemplateRecognizer
            template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character")
            if os.path.exists(template_dir):
                self.template_recognizer = TemplateRecognizer(template_dir, use_deep_feature=True)
                print(f"✓ HOG 模板识别器已加载: {template_dir}")
            else:
                print(f"✗ 模板目录不存在: {template_dir}")
        except Exception as e:
            print(f"✗ HOG 识别器加载失败: {e}")
    
    def decode_lprnet(self, preds):
        """解码 LPRNet 输出"""
        pred_labels = []
        for i in range(preds.size(0)):
            pred = preds[i]
            pred_indices = torch.argmax(pred, dim=1)
            char_list = []
            prev_idx = -1
            for idx in pred_indices:
                idx = idx.item()
                if idx != prev_idx and idx != len(self.CHARS)-1:
                    char_list.append(self.CHARS[idx])
                prev_idx = idx
            pred_labels.append("".join(char_list))
        return pred_labels
    
    def recognize_lprnet(self, warped):
        """使用 LPRNet 识别"""
        if self.lpr_model is None:
            return None, 0.0
        
        # LPRNet 输入: 94x24
        lpr_input = cv2.resize(warped, (94, 24))
        lpr_input = lpr_input.astype('float32') / 255.0
        lpr_input = lpr_input.transpose(2, 0, 1)
        lpr_input = torch.tensor(lpr_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.lpr_model(lpr_input)
            plate_text = self.decode_lprnet(output)[0]
        
        return plate_text, 1.0  # LPRNet 不直接输出置信度
    
    def recognize_hog(self, warped, method='cosine'):
        """使用 HOG 模板匹配识别"""
        if self.template_recognizer is None:
            return None, 0.0
        
        # 设置匹配方法
        if method == 'euclidean':
            self.template_recognizer.set_match_method('euclidean')
        else:
            self.template_recognizer.set_match_method('cosine')
        
        result = self.template_recognizer.recognize_with_details(warped, plate_type='140')
        return result['plate_number'], result['confidence']
    
    def test_single_image(self, image_path, methods=['lprnet', 'hog_euclidean', 'hog_cosine']):
        """测试单张图片"""
        # 解析文件名获取真实车牌号
        info = parse_ccpd_filename(image_path)
        if info is None:
            return None
        
        # 读取图片
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        
        # 透视变换
        warped = four_point_transform(img, info['vertices'])
        
        results = {
            'ground_truth': info['plate_number'],
            'predictions': {}
        }
        
        for method in methods:
            if method == 'lprnet':
                pred, conf = self.recognize_lprnet(warped)
            elif method == 'hog_euclidean':
                pred, conf = self.recognize_hog(warped, 'euclidean')
            elif method == 'hog_cosine':
                pred, conf = self.recognize_hog(warped, 'cosine')
            else:
                continue
            
            if pred is not None:
                results['predictions'][method] = {
                    'plate_number': pred,
                    'confidence': conf,
                    'correct': pred == info['plate_number']
                }
        
        return results
    
    def test_dataset(self, ccpd_root, max_images=None, subset='ccpd_base'):
        """
        测试 CCPD 数据集
        
        Args:
            ccpd_root: CCPD 数据集根目录
            max_images: 最大测试图片数（None 表示全部）
            subset: 测试子集 ('ccpd_base', 'ccpd_blur', 'ccpd_db', 等)
        """
        # 收集图片
        test_dir = os.path.join(ccpd_root, subset)
        if not os.path.exists(test_dir):
            # 尝试直接使用 ccpd_root
            test_dir = ccpd_root
        
        image_files = []
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.endswith('.jpg'):
                    image_files.append(os.path.join(root, f))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n测试集: {test_dir}")
        print(f"图片数量: {len(image_files)}")
        
        methods = ['lprnet', 'hog_euclidean', 'hog_cosine']
        stats = {m: {'correct': 0, 'total': 0, 'char_correct': 0, 'char_total': 0, 'time': 0} for m in methods}
        
        failed_samples = defaultdict(list)
        
        for img_path in tqdm(image_files, desc="测试中"):
            results = self.test_single_image(img_path, methods)
            if results is None:
                continue
            
            gt = results['ground_truth']
            
            for method, pred_info in results['predictions'].items():
                stats[method]['total'] += 1
                stats[method]['char_total'] += len(gt)
                
                pred = pred_info['plate_number']
                if pred_info['correct']:
                    stats[method]['correct'] += 1
                    stats[method]['char_correct'] += len(gt)
                else:
                    # 计算字符级别正确率
                    for i, (g, p) in enumerate(zip(gt, pred)):
                        if g == p:
                            stats[method]['char_correct'] += 1
                    
                    # 记录错误样本
                    if len(failed_samples[method]) < 10:
                        failed_samples[method].append({
                            'file': os.path.basename(img_path),
                            'gt': gt,
                            'pred': pred
                        })
        
        # 打印结果
        print("\n" + "="*70)
        print("测试结果汇总")
        print("="*70)
        print(f"{'方法':<20} {'车牌准确率':<15} {'字符准确率':<15} {'测试数量':<10}")
        print("-"*70)
        
        for method in methods:
            s = stats[method]
            if s['total'] > 0:
                plate_acc = s['correct'] / s['total'] * 100
                char_acc = s['char_correct'] / s['char_total'] * 100 if s['char_total'] > 0 else 0
                print(f"{method:<20} {plate_acc:>10.2f}%     {char_acc:>10.2f}%     {s['total']:<10}")
        
        print("="*70)
        
        # 打印部分错误样本
        print("\n错误样本示例:")
        for method in methods:
            if failed_samples[method]:
                print(f"\n[{method}]")
                for sample in failed_samples[method][:5]:
                    print(f"  文件: {sample['file'][:30]}...")
                    print(f"  真实: {sample['gt']} -> 预测: {sample['pred']}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='CCPD 数据集测试')
    parser.add_argument('--ccpd_root', type=str, default='./CCPD2019',
                        help='CCPD 数据集根目录')
    parser.add_argument('--subset', type=str, default='ccpd_base',
                        help='测试子集 (ccpd_base, ccpd_blur, ccpd_db, ccpd_fn, ccpd_rotate, ccpd_tilt, ccpd_challenge)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='最大测试图片数（用于快速测试）')
    parser.add_argument('--single', type=str, default=None,
                        help='测试单张图片')
    
    args = parser.parse_args()
    
    tester = CCPDTester()
    
    if args.single:
        # 测试单张图片
        if os.path.exists(args.single):
            results = tester.test_single_image(args.single)
            if results:
                print(f"\n真实车牌: {results['ground_truth']}")
                for method, info in results['predictions'].items():
                    status = "✓" if info['correct'] else "✗"
                    print(f"{method}: {info['plate_number']} ({info['confidence']:.2%}) {status}")
        else:
            print(f"文件不存在: {args.single}")
    else:
        # 测试数据集
        if os.path.exists(args.ccpd_root):
            tester.test_dataset(args.ccpd_root, args.max_images, args.subset)
        else:
            print(f"CCPD 数据集目录不存在: {args.ccpd_root}")
            print("\n请下载 CCPD 数据集:")
            print("  Google Drive: https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc")
            print("  百度网盘: https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw (提取码: hm0u)")
            print("\n使用方法:")
            print("  python test_ccpd.py --ccpd_root ./CCPD2019 --subset ccpd_base --max_images 100")


if __name__ == "__main__":
    main()
