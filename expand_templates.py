"""
字符模板库扩充脚本
从 CCPD 数据集中提取字符样本，扩充模板库

安全策略：
1. 只有当分割字符数 == 真实车牌字符数时才进行扩充
2. 每个字符保存多个变体，文件名格式: X_1.jpg, X_2.jpg, ...
3. 可以预览待添加的字符，确认后再保存
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict

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
    """解析 CCPD 文件名"""
    try:
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('-')
        if len(parts) < 5:
            return None
        
        plate_indices = parts[4].split('_')
        if len(plate_indices) < 7:
            return None
            
        plate_str = PROVINCES[int(plate_indices[0])]
        plate_str += ALPHABETS[int(plate_indices[1])]
        for idx in plate_indices[2:]:
            plate_str += ADS[int(idx)]
        
        # 解析四个顶点
        vertices_str = parts[3]
        v_coords = vertices_str.split('_')
        vertices = []
        for v in v_coords:
            vx, vy = v.split('&')
            vertices.append([int(vx), int(vy)])
        
        vertices = np.array([vertices[2], vertices[3], vertices[0], vertices[1]], dtype="float32")
        
        return {
            'plate_number': plate_str,
            'vertices': vertices,
        }
    except:
        return None


def four_point_transform(image, pts, width=240, height=80):
    """透视变换"""
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def get_next_template_index(char_dir, char_name):
    """获取下一个模板编号"""
    existing = []
    for f in os.listdir(char_dir):
        if f.startswith(char_name) and f.endswith(('.png', '.jpg', '.jpeg')):
            # 解析编号: X.png -> 0, X_1.png -> 1, X_2.jpg -> 2
            name = os.path.splitext(f)[0]
            if name == char_name:
                existing.append(0)
            elif name.startswith(char_name + '_'):
                try:
                    idx = int(name.split('_')[1])
                    existing.append(idx)
                except:
                    pass
    
    if not existing:
        return 0
    return max(existing) + 1


def expand_templates(ccpd_root, template_dir, max_images=100, max_per_char=5, 
                     preview=True, auto_save=False):
    """
    从 CCPD 数据集扩充字符模板库
    
    Args:
        ccpd_root: CCPD 图片目录
        template_dir: 模板库目录
        max_images: 最多处理的图片数
        max_per_char: 每个字符最多添加的模板数
        preview: 是否预览
        auto_save: 是否自动保存（不预览）
    """
    from template_match.segment_v3 import segment_with_fallback
    
    chinese_dir = os.path.join(template_dir, 'chinese')
    alpha_dir = os.path.join(template_dir, 'alphanumeric')
    os.makedirs(chinese_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)
    
    # 收集图片
    image_files = []
    for root, _, files in os.walk(ccpd_root):
        for f in files:
            if f.endswith('.jpg'):
                image_files.append(os.path.join(root, f))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"处理图片数: {len(image_files)}")
    
    # 统计
    stats = {
        'total': 0,
        'matched': 0,  # 字符数匹配
        'mismatched': 0,  # 字符数不匹配
        'added': defaultdict(int),  # 每个字符添加的数量
    }
    
    # 待添加的字符
    pending_chars = defaultdict(list)  # {char: [(img, source_file), ...]}
    
    for img_path in tqdm(image_files, desc="分析中"):
        info = parse_ccpd_filename(img_path)
        if info is None:
            continue
        
        stats['total'] += 1
        gt = info['plate_number']
        
        # 读取并矫正
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        warped = four_point_transform(img, info['vertices'])
        
        # 分割字符
        char_images = segment_with_fallback(warped, len(gt))
        
        # 关键检查：字符数必须匹配
        if len(char_images) != len(gt):
            stats['mismatched'] += 1
            continue
        
        stats['matched'] += 1
        
        # 收集字符
        for i, (char_img, char_label) in enumerate(zip(char_images, gt)):
            # 检查是否需要更多模板
            if i == 0:
                # 汉字
                char_dir = chinese_dir
            else:
                # 字母数字
                char_dir = alpha_dir
            
            # 检查当前模板数量
            current_count = len([f for f in os.listdir(char_dir) 
                               if f.startswith(char_label) and f.endswith(('.png', '.jpg', '.jpeg'))])
            
            if current_count < max_per_char:
                # 还需要更多模板
                if len(pending_chars[char_label]) < max_per_char - current_count:
                    pending_chars[char_label].append((char_img, os.path.basename(img_path)))
    
    print(f"\n统计:")
    print(f"  总图片: {stats['total']}")
    print(f"  字符数匹配: {stats['matched']} ({stats['matched']/max(stats['total'],1)*100:.1f}%)")
    print(f"  字符数不匹配: {stats['mismatched']}")
    print(f"  待添加字符种类: {len(pending_chars)}")
    
    if not pending_chars:
        print("\n没有需要添加的新模板")
        return
    
    # 显示待添加的字符
    print(f"\n待添加的字符模板:")
    for char, samples in sorted(pending_chars.items()):
        print(f"  '{char}': {len(samples)} 个样本")
    
    if preview and not auto_save:
        # 预览模式：显示每个字符的样本
        print("\n预览模式 - 按 's' 保存当前字符，'n' 跳过，'q' 退出")
        
        for char, samples in sorted(pending_chars.items()):
            if not samples:
                continue
            
            # 创建预览图
            preview_imgs = []
            for char_img, source in samples:
                # 确保是彩色图
                if len(char_img.shape) == 2:
                    char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
                
                # 调整大小便于显示
                h, w = char_img.shape[:2]
                scale = 80 / h
                resized = cv2.resize(char_img, (int(w * scale), 80))
                
                # 添加边框
                bordered = cv2.copyMakeBorder(resized, 2, 2, 2, 2, 
                                             cv2.BORDER_CONSTANT, value=(0, 255, 0))
                preview_imgs.append(bordered)
            
            # 拼接
            max_w = max(img.shape[1] for img in preview_imgs)
            padded = []
            for img in preview_imgs:
                # 确保是3通道
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if img.shape[1] < max_w:
                    pad = np.ones((img.shape[0], max_w - img.shape[1], 3), dtype=np.uint8) * 255
                    img = np.hstack([img, pad])
                padded.append(img)
            
            combined = np.vstack(padded) if len(padded) > 1 else padded[0]
            
            # 确定保存目录
            if char in PROVINCES:
                char_dir = chinese_dir
            else:
                char_dir = alpha_dir
            
            # 显示预览窗口
            window_name = f"Char '{char}' - {len(samples)} samples"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined)
            
            print(f"\n当前字符: '{char}' ({len(samples)} 个样本)")
            print("  按 's' 保存, 'n' 跳过, 'q' 退出 (请先点击图片窗口)")
            
            # 等待按键，超时后也继续
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('s') or key == ord('S'):
                    # 保存
                    for char_img, source in samples:
                        idx = get_next_template_index(char_dir, char)
                        if idx == 0:
                            save_name = f"{char}.jpg"
                        else:
                            save_name = f"{char}_{idx}.jpg"
                        save_path = os.path.join(char_dir, save_name)
                        cv2.imencode('.jpg', char_img)[1].tofile(save_path)
                        stats['added'][char] += 1
                    print(f"  ✓ 已保存 '{char}': {len(samples)} 个模板")
                    break
                elif key == ord('n') or key == ord('N'):
                    print(f"  - 跳过 '{char}'")
                    break
                elif key == ord('q') or key == ord('Q'):
                    print("退出预览")
                    cv2.destroyAllWindows()
                    return
                elif key == 27:  # ESC
                    print("退出预览")
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyAllWindows()
    
    elif auto_save:
        # 自动保存模式
        print("\n自动保存模式...")
        for char, samples in pending_chars.items():
            if char in PROVINCES:
                char_dir = chinese_dir
            else:
                char_dir = alpha_dir
            
            for char_img, source in samples:
                idx = get_next_template_index(char_dir, char)
                if idx == 0:
                    save_name = f"{char}.jpg"
                else:
                    save_name = f"{char}_{idx}.jpg"
                save_path = os.path.join(char_dir, save_name)
                cv2.imencode('.jpg', char_img)[1].tofile(save_path)
                stats['added'][char] += 1
        
        print(f"已保存 {sum(stats['added'].values())} 个新模板")
    
    # 最终统计
    if stats['added']:
        print(f"\n添加的模板:")
        for char, count in sorted(stats['added'].items()):
            print(f"  '{char}': +{count}")


def main():
    parser = argparse.ArgumentParser(description='从 CCPD 扩充字符模板库')
    parser.add_argument('--ccpd_root', type=str, required=True,
                        help='CCPD 图片目录')
    parser.add_argument('--template_dir', type=str, default='./character',
                        help='模板库目录')
    parser.add_argument('--max_images', type=int, default=100,
                        help='最多处理的图片数')
    parser.add_argument('--max_per_char', type=int, default=5,
                        help='每个字符最多保留的模板数')
    parser.add_argument('--auto', action='store_true',
                        help='自动保存模式（不预览）')
    parser.add_argument('--no_preview', action='store_true',
                        help='只分析不保存')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ccpd_root):
        print(f"目录不存在: {args.ccpd_root}")
        return
    
    expand_templates(
        args.ccpd_root,
        args.template_dir,
        max_images=args.max_images,
        max_per_char=args.max_per_char,
        preview=not args.no_preview,
        auto_save=args.auto
    )


if __name__ == "__main__":
    main()
