import os




ANCHOR_SIZE_THRESHOLD = 480.0 * 270.0 * 0.0025

OUTPUT_ROOT_ORIGIN = '/home/user/frog/mot-dbt/tracking_wo_bnw/output/tracktor/MOT17/tracktor-nofovea-0820'
OUTPUT_ROOT_FOVEA = '/home/user/frog/mot-dbt/tracking_wo_bnw/output/tracktor/MOT17/tracktor-fovea-0820'
MOT17_DIR = '/home/user/frog/mot-dbt/tracking_wo_bnw/data/MOT17-down4.0/train'
RESULT_OUTPUT_DIR = '/home/user/frog/mot-dbt/tracking_wo_bnw/output/frog_debug/0820_match'


def read_match_file(match_file_path):
    """读取match.txt文件,返回一个字典,键是帧编号,值是这一帧匹配上的锚框列表"""
    match_data = {}
    with open(match_file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            frame_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            if frame_id not in match_data:
                match_data[frame_id] = []
            match_data[frame_id].append(bbox)
    return match_data


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


def count_small_anchors_for_match(match_data, anchor_size_threshold):
    """统计每一帧中小于阈值的锚框数量"""
    small_anchors_count = {}
    for frame_id, bboxes in match_data.items():
        count = 0
        for bbox in bboxes:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w * h < anchor_size_threshold:
                count += 1
        small_anchors_count[frame_id] = count
    return small_anchors_count


def count_small_anchors_for_gt(gt_data, anchor_size_threshold):
    """统计每一帧中小于阈值的gt数量"""
    small_gt_count = {}
    for frame_id, bboxes in gt_data.items():
        count = 0
        for bbox in bboxes:
            w = bbox['bbox'][2] - bbox['bbox'][0]
            h = bbox['bbox'][3] - bbox['bbox'][1]
            if w * h < anchor_size_threshold:
                count += 1
        small_gt_count[frame_id] = count
    return small_gt_count


if __name__ == '__main__':
    sequences = [
        'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
        'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'
    ]

    for seq in sequences:
        match_file_path_fovea = os.path.join(OUTPUT_ROOT_FOVEA, seq, 'match.txt')
        match_file_path_origin = os.path.join(OUTPUT_ROOT_ORIGIN, seq, 'match.txt')
        gt_file_path = os.path.join(MOT17_DIR, seq, 'gt', 'gt.txt')
        match_data_fovea = read_match_file(match_file_path_fovea)
        match_data_origin = read_match_file(match_file_path_origin)
        gt_data = read_gt_file(gt_file_path)

        small_anchor_fovea = count_small_anchors_for_match(match_data_fovea, ANCHOR_SIZE_THRESHOLD)
        small_anchors_origin = count_small_anchors_for_match(match_data_origin, ANCHOR_SIZE_THRESHOLD)
        small_gt = count_small_anchors_for_gt(gt_data, ANCHOR_SIZE_THRESHOLD)

        small_gt_count = 0
        for frame_id in small_gt:
            small_gt_count += small_gt[frame_id]
        
        small_anchor_fovea_count, small_anchor_origin_count = 0, 0
        for frame_id in small_anchor_fovea:
            small_anchor_fovea_count += small_anchor_fovea[frame_id]
        
        for frame_id in small_anchors_origin:
            small_anchor_origin_count += small_anchors_origin[frame_id]
        
        print(f'{seq}:')
        print(f'小于阈值的gt数量: {small_gt_count}')
        print(f'fovea小于阈值的锚框数量: {small_anchor_fovea_count}')
        print(f'origin小于阈值的锚框数量: {small_anchor_origin_count}')
        