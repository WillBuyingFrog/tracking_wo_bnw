import os
import cv2
import numpy as np
import argparse


OUTPUT_ROOT_ORIGIN = '/home/user/frog/mot-dbt/tracking_wo_bnw/output/tracktor/MOT17/tracktor-nofovea-0816'
OUTPUT_ROOT_FOVEA = '/home/user/frog/mot-dbt/tracking_wo_bnw/output/tracktor/MOT17/tracktor-fovea-0901'
MOT17_DIR = '/home/user/frog/mot-dbt/tracking_wo_bnw/data/MOT17-down4.0/train'
FP_OUTPUT_DIR = '/data/frog/2409/mot-dbt-debug/output/0901_fp'


def read_fp_file(fp_file_path):
    """读取fp.txt文件，返回一个字典，键是帧编号，值是锚框列表"""
    fp_data = {}
    with open(fp_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            frame_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            if frame_id not in fp_data:
                fp_data[frame_id] = []
            fp_data[frame_id].append(bbox)
    return fp_data

def read_gt_file(gt_file_path, frame_start=0, frame_end=-1):
    gt_data = {}
    with open(gt_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            if frame_id < frame_start:
                continue
            if frame_end != -1 and frame_id > frame_end:
                break
            if frame_id not in gt_data:
                gt_data[frame_id] = []
            bbox = list(map(float, parts[2:6]))
            # 转换成tlbr格式
            bbox[2] += bbox[0]  
            bbox[3] += bbox[1]
            conf = int(parts[6])
            class_name = int(parts[7])
            visibility = float(parts[8])
            single_data = {
                'bbox': bbox,
                'conf': conf,
                'class_name': class_name,
                'visibility': visibility
            }
            gt_data[frame_id].append(single_data)

    return gt_data

def calculate_iou_fp_gt(gt_data, fp_bbox, frame_id, 
                        all_classes=False, output_path='test', seq_name='test'):
    """计算指定False Positive Case与Ground Truth的IOU"""
    print(f"Unique FP bbox {fp_bbox} ious with gt bboxes")
    for gt_bbox in gt_data[frame_id]:
        if not all_classes and gt_bbox['class_name'] != 1:
            continue
        if not all_classes and gt_bbox['conf'] < 1:
            continue
        iou = calculate_iou(gt_bbox['bbox'], fp_bbox)        
        print(f"\t IOU with bbox {gt_bbox}: {iou}")
            
def calculate_iou_fp_gt_all_frames(gt_data, unique_fp_bbox,
                        all_classes=False, output_path='test', seq_name='test'):
    """计算指定sequence中所有False Positive case与Ground Truth的IOU
    """
    intersect_gt_bbox = {}
    intersect_gt_bbox_iou = {}
    with open(os.path.join(output_path, f"unique_fps_iou_{seq_name}.txt"), "w") as iou_output_file:
        for frame_id in unique_fp_bbox:
            iou_output_file.write(f"At frame {frame_id}\n")
            intersect_gt_bbox[frame_id] = []
            intersect_gt_bbox_iou[frame_id] = []
            for fp_bbox in unique_fp_bbox[frame_id]:
                iou_output_file.write(f"\tUnique FP bbox {fp_bbox} ious with gt bboxes:\n")
                gt_frame_id = frame_id + 1
                for gt_bbox in gt_data[gt_frame_id]:
                    if not all_classes and gt_bbox['class_name'] != 1:
                        continue
                    if not all_classes and gt_bbox['conf'] < 1:
                        continue
                    iou = calculate_iou(gt_bbox['bbox'], fp_bbox)
                    # if iou > 0.1 and iou < 0.5:
                    if iou > 0.1:
                        intersect_gt_bbox[frame_id].append(gt_bbox)
                        intersect_gt_bbox_iou[frame_id].append(iou)
                        iou_output_file.write(f"\t\t{frame_id} {gt_bbox['bbox'][0]} {gt_bbox['bbox'][1]} {gt_bbox['bbox'][2]} {gt_bbox['bbox'][3]} {iou}\n")
        
    return intersect_gt_bbox, intersect_gt_bbox_iou


def calculate_iou(bbox1, bbox2):
    """计算两个锚框的IOU"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    # 计算交集的坐标
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    # 计算交集面积
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    # 计算两个锚框的面积
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    # 计算并集面积
    union_area = bbox1_area + bbox2_area - inter_area
    # 计算IOU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def find_unique_fps(fp_data_a, fp_data_b):
    """对A、B方法而言，找出A方法在同一帧中独有的False Positive Case"""
    unique_fps_a = {}
    for frame_id in fp_data_a:
        if frame_id not in fp_data_b:
            # print(f"No false positive cases for method b at frame {frame_id}")
            unique_fps_a[frame_id] = fp_data_a[frame_id]
        else:
            unique_fps_a[frame_id] = []
            for bbox_a in fp_data_a[frame_id]:
                found = False
                for bbox_b in fp_data_b[frame_id]:
                    if calculate_iou(bbox_a, bbox_b) > 0.5:
                        # if frame_id == 0:
                        #     print(f"Found a match at frame {frame_id}, {bbox_a} {bbox_b}, iou={calculate_iou(bbox_a, bbox_b)}")
                        found = True
                        break
                if not found:
                    # if frame_id == 0:
                    #     print(f"Found a unique false positive case at frame {frame_id}, {bbox_a}")
                    unique_fps_a[frame_id].append(bbox_a)
    return unique_fps_a

def draw_bboxes(image, bboxes, color=(0, 0, 255)):
    """在图像上绘制unique false positive case的锚框"""
    for bbox in bboxes:
        x, y, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x, y), (x2, y2), color, 1)
    return image

def draw_gt_bboxes(image, gt_bboxes, color=(0, 255, 0)):
    """在图像上绘制gt的锚框"""
    for bbox in gt_bboxes:
        x, y, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x, y), (x2, y2), color, 1)
    return image

def process_sequence(sequence_dir_a, sequence_dir_b, mot17_dir, output_dir):
    """处理一个sequence，画出独有False Positive Case的锚框并保存"""

    sequence_name = os.path.basename(sequence_dir_b)
    sequence_output_dir = os.path.join(output_dir, sequence_name)

    print(f"Processing sequence {sequence_name}")

    fp_file_a = os.path.join(sequence_dir_a, 'fp.txt')
    fp_file_b = os.path.join(sequence_dir_b, 'fp.txt')
    fp_data_a = read_fp_file(fp_file_a)
    fp_data_b = read_fp_file(fp_file_b)
    unique_fps_a = find_unique_fps(fp_data_a, fp_data_b)
    
    os.makedirs(sequence_output_dir, exist_ok=True)
    os.makedirs(os.path.join(sequence_output_dir, "gt"), exist_ok=True)

    gt_path = os.path.join(mot17_dir, sequence_name, 'gt', 'gt.txt')
    gt_data = read_gt_file(gt_path)
    # if 0 in unique_fps_a and sequence_name == 'MOT17-02-FRCNN':
    #     for bboxes in unique_fps_a[0]:
    #         calculate_iou_fp_gt(gt_data, bboxes, 1, all_classes=True)
    intersect_gt_bbox, intersect_gt_bbox_iou = calculate_iou_fp_gt_all_frames(gt_data, unique_fps_a, all_classes=False, output_path=output_dir, seq_name=sequence_name)

    with open(os.path.join(output_dir, f'unique_fps_{sequence_name}.txt'), 'w') as fp_output_file:
        for frame_id in unique_fps_a:
            output_frame_id = frame_id + 1
            image_path = os.path.join(mot17_dir, sequence_name, 'img1', f'{output_frame_id:06d}.jpg')
            if not os.path.exists(image_path):
                continue
            image = cv2.imread(image_path)
            image = draw_bboxes(image, unique_fps_a[frame_id])
            output_path = os.path.join(sequence_output_dir, f'{output_frame_id:06d}.jpg')
            cv2.imwrite(output_path, image)
            # 如果有符合绘制条件的gt bbox,那么再绘制该帧的gt bbox
            if len(intersect_gt_bbox[frame_id]) > 0:
                image = draw_gt_bboxes(image,[gt['bbox'] for gt in intersect_gt_bbox[frame_id]])
                output_path = os.path.join(sequence_output_dir, "gt", f'{output_frame_id:06d}_gt.jpg')
                cv2.imwrite(output_path, image)
            for unique_box  in unique_fps_a[frame_id]:
                fp_output_file.write(f"{frame_id} {unique_box[0]} {unique_box[1]} {unique_box[2]} {unique_box[3]}\n")
            
            
    



def main(output_root_origin, output_root_fovea, mot17_dir, fp_output_dir):
    """主函数,处理所有sequence"""
    sequences = [
        'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
        'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'
    ]
    for sequence in sequences:
        sequence_dir_a = os.path.join(output_root_fovea, sequence)
        sequence_dir_b = os.path.join(output_root_origin, sequence)
        process_sequence(sequence_dir_a, sequence_dir_b, mot17_dir, fp_output_dir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root_origin', type=str, default=OUTPUT_ROOT_ORIGIN)
    parser.add_argument('--output_root_fovea', type=str, default=OUTPUT_ROOT_FOVEA)
    parser.add_argument('--mot17_dir', type=str, default=MOT17_DIR)
    parser.add_argument('--fp_output_dir', type=str, default=FP_OUTPUT_DIR)
    args = parser.parse_args()

    main(args.output_root_origin, args.output_root_fovea, args.mot17_dir, args.fp_output_dir)

    # 示例调用
    # main(OUTPUT_ROOT_ORIGIN, OUTPUT_ROOT_FOVEA, MOT17_DIR, FP_OUTPUT_DIR)