
import os
import cv2
import argparse

def create_downsample_mot17_structure(target_root, origin_root):

    # 遍历origin_root中，train文件夹和test文件夹内的所有MOT17数据，将每个MOT17数据的文件夹结构复制到target_root中
    for phase in ['train', 'test']:
        for seq in os.listdir(os.path.join(origin_root, phase)):
            if seq.startswith('MOT17-'):
                target_seq_path = os.path.join(target_root, phase, seq)
                if not os.path.exists(target_seq_path):
                    os.makedirs(target_seq_path)
                # 如果是train文件夹下的sequence，则需要创建img1, det, gt;test下的sequence时没有gt的。
                if phase == 'train':
                    for sub_folder in ['img1', 'det', 'gt']:
                        sub_folder_path = os.path.join(target_seq_path, sub_folder)
                        if not os.path.exists(sub_folder_path):
                            os.makedirs(sub_folder_path)
                else:
                    for sub_folder in ['img1', 'det']:
                        sub_folder_path = os.path.join(target_seq_path, sub_folder)
                        if not os.path.exists(sub_folder_path):
                            os.makedirs(sub_folder_path)




def downsample_mot17_sequence(origin_seq_root, target_seq_root,
                               downsample_ratio=3.0, txt_only=False):

    # 第一步：将origin_seq_root代表的MOT17 sequence中的图片，长宽各压缩downsample_ratio倍后，保存到target_seq_root中
    # 保证origin_seq_root和target_seq_root下的文件夹结构完全一致，即origin_seq_root下，以MOT17格式组织的img1, gt（这两个一定有）, det（如果有）文件夹结构一定存在
    img1_folder = os.path.join(origin_seq_root, 'img1')
    target_img1_folder = os.path.join(target_seq_root, 'img1')
    if txt_only:
        print(f'Only rewriting txt file, skipping images')
    else:
        for img_name in os.listdir(img1_folder):
            img_path = os.path.join(img1_folder, img_name)
            target_img_path = os.path.join(target_img1_folder, img_name)
            # 读取图片，将图片压缩downsample_ratio倍
            img = cv2.imread(img_path)
            img = cv2.resize(img, (int(img.shape[1] / downsample_ratio), int(img.shape[0] / downsample_ratio)))
            cv2.imwrite(target_img_path, img)

    # 第二步：将origin_seq_root代表的MOT17 sequence中的gt文件夹中的gt文件，压缩downsample_ratio倍后，保存到target_seq_root中

    # 鉴于test sequence下是没有gt的，因此需要判断gt文件夹是否存在
    if os.path.exists(os.path.join(origin_seq_root, 'gt')):
        gt_folder = os.path.join(origin_seq_root, 'gt')
        target_gt_folder = os.path.join(target_seq_root, 'gt')
        for gt_name in os.listdir(gt_folder):
            gt_path = os.path.join(gt_folder, gt_name)
            target_gt_path = os.path.join(target_gt_folder, gt_name)
            with open(gt_path, 'r') as f:
                lines = f.readlines()
            with open(target_gt_path, 'w') as f:
                for line in lines:
                    line = line.strip().split(',')
                    # 对坐标的downsample结果都保留int
                    line[2] = str(int(int(line[2]) / downsample_ratio))
                    line[3] = str(int(int(line[3]) / downsample_ratio))
                    line[4] = str(int(int(line[4]) / downsample_ratio))
                    line[5] = str(int(int(line[5]) / downsample_ratio))
                    f.write(','.join(line) + '\n')

    # 第三步：将origin_seq_root代表的MOT17 sequence中的det文件夹中的det文件，压缩downsample_ratio倍后，保存到target_seq_root中
    # 这里默认det文件是txt格式的，每行是一个检测结果，格式为：frame_id, id, x, y, w, h, confidence, class, visibility

    det_folder = os.path.join(origin_seq_root, 'det')
    target_det_folder = os.path.join(target_seq_root, 'det')
    for det_name in os.listdir(det_folder):
        det_path = os.path.join(det_folder, det_name)
        target_det_path = os.path.join(target_det_folder, det_name)
        with open(det_path, 'r') as f:
            lines = f.readlines()
        with open(target_det_path, 'w') as f:
            for line in lines:
                line = line.strip().split(',')
                line[2] = str(round(float(line[2]) / downsample_ratio, 1))
                line[3] = str(round(float(line[3]) / downsample_ratio, 1))
                line[4] = str(round(float(line[4]) / downsample_ratio, 1))
                line[5] = str(round(float(line[5]) / downsample_ratio, 1))
                f.write(','.join(line) + '\n')

    # 第四步：创建目标文件夹下的seqinfo
    # 除了图像长宽需要改，剩下的都不需要
    seqinfo_path = os.path.join(origin_seq_root, 'seqinfo.ini')
    target_seqinfo_path = os.path.join(target_seq_root, 'seqinfo.ini')
    print(f'Writing seqinfo to {target_seqinfo_path}')
    with open(seqinfo_path, 'r') as f:
        lines = f.readlines()
        with open(target_seqinfo_path, 'w') as f1:
            for line in lines:
                if line.startswith('imWidth'):
                    imWidth = int(line.strip().split('=')[1])
                    f1.write('imWidth=' + str(int(imWidth / downsample_ratio)) + '\n')
                elif line.startswith('imHeight'):
                    imHeight = int(line.strip().split('=')[1])
                    f1.write('imHeight=' + str(int(imHeight / downsample_ratio)) + '\n')
                else:
                    f1.write(line)
    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_root', type=str, default='/root/tracking_wo_bnw/data/MOT17')
    parser.add_argument('--target_root', type=str, default='/root/tracking_wo_bnw/data/MOT17-down4.0')
    parser.add_argument('--downsample_ratio', type=float, default=4.0)
    parser.add_argument('--txt_only', action='store_true')

    args = parser.parse_args()

    # 先创建文件结构
    create_downsample_mot17_structure(args.target_root, args.origin_root)

    # 再downsample每个sequence
    for phase in ['train', 'test']:
        for seq in os.listdir(os.path.join(args.origin_root, phase)):
            if seq.startswith('MOT17-'):
                origin_seq_root = os.path.join(args.origin_root, phase, seq)
                target_seq_root = os.path.join(args.target_root, phase, seq)
                downsample_mot17_sequence(origin_seq_root, target_seq_root,
                                           downsample_ratio=args.downsample_ratio, txt_only=args.txt_only)
                print('Downsampled sequence:', origin_seq_root, '->', target_seq_root)