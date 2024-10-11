import copy
import os
import time
from os import path as osp
import argparse

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
import random
from sacred import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.oracle_tracker import OracleTracker
from tracktor.reid.resnet import ReIDNetwork_resnet50
from tracktor.tracker import Tracker
from tracktor.utils import (evaluate_mot_accums, get_mot_accum,
                            interpolate_tracks, plot_sequence)
from tracktor.reid.config import (check_cfg, engine_run_kwargs,
                                  get_default_config, lr_scheduler_kwargs,
                                  optimizer_kwargs, reset_config)
from torchreid.utils import FeatureExtractor

from PIL import Image


mm.lap.default_solver = 'lap'

CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1010_01.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1007_01.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1006_04.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1006_03.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1006_02.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1006_01.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1004_03.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1004_02.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_1004.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0917.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0914.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0912.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0911.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0910.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0903.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_fovea_0901.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_origin.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_origin_down4.0.yaml'
# CONFIG_FILE = 'experiments/cfgs/tracktor_origin_down3.0.yaml'

ex = Experiment()



ex.add_config(CONFIG_FILE)
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')



# @ex.config
def add_reid_config(reid_models, obj_detect_models, dataset):
    # if isinstance(dataset, str):
    #     dataset = [dataset]
    if isinstance(reid_models, str):
        reid_models = [reid_models, ] * len(dataset)

    # if multiple reid models are provided each is applied
    # to a different dataset
    if len(reid_models) > 1:
        assert len(dataset) == len(reid_models)

    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)

    return reid_models, obj_detect_models, dataset


@ex.automain
def main(module_name, name, seed, obj_detect_models, reid_models,
         tracker, oracle, dataset, load_results, frame_range, interpolate,
         write_images, _config, _log, _run):
    
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    output_dir = osp.join(get_output_dir(module_name), name)
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(copy.deepcopy(_config), outfile, default_flow_style=False)

    dataset = Datasets(dataset)
    reid_models, obj_detect_models, dataset = add_reid_config(reid_models, obj_detect_models, dataset)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector(s).")

    obj_detects = []
    origin_obj_detects = []
    for obj_detect_model in obj_detect_models:
        obj_detect = FRCNN_FPN(num_classes=2)
        origin_obj_detect = FRCNN_FPN(num_classes=2)

        obj_detect_state_dict = torch.load(
            obj_detect_model, map_location=lambda storage, loc: storage)
        if 'model' in obj_detect_state_dict:
            obj_detect_state_dict = obj_detect_state_dict['model']

        obj_detect.load_state_dict(obj_detect_state_dict)

        # 0909修改
        try:
            FOVEA_OBJ_DETECT_MODEL_PATH = tracker['frog_fovea_obj_detect_model_path']
        except:
            print("No fovea obj detect model path found, use default path")
            FOVEA_OBJ_DETECT_MODEL_PATH = 'output/faster_rcnn_fpn_training_mot_17/fovea_0910_03_model_epoch_20.model'
        print(f"Loading fovea obj detect model from {FOVEA_OBJ_DETECT_MODEL_PATH}")
        origin_obj_detect_state_dict = torch.load(FOVEA_OBJ_DETECT_MODEL_PATH, map_location=lambda storage, loc: storage)

        origin_obj_detect.load_state_dict(origin_obj_detect_state_dict)

        obj_detects.append(obj_detect)
        origin_obj_detects.append(origin_obj_detect)

        obj_detect.eval()
        origin_obj_detect.eval()
        if torch.cuda.is_available():
            obj_detect.cuda()
            origin_obj_detect.cuda()

    # reid
    _log.info("Initializing reID network(s).")

    reid_networks = []
    for reid_model in reid_models:
        assert os.path.isfile(reid_model)
        reid_network = FeatureExtractor(
            model_name='resnet50_fc512',
            model_path=reid_model,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        reid_networks.append(reid_network)
    # tracktor
    if oracle is not None:
        tracker = OracleTracker(
            obj_detect, reid_network, tracker, oracle)
    else:
        tracker = Tracker(obj_detect, reid_network, tracker,
                          origin_obj_detect=origin_obj_detect)

    time_total = 0
    num_frames = 0
    mot_accums = []

    for seq, obj_detect, origin_obj_detect, reid_network in zip(dataset, obj_detects, origin_obj_detects, reid_networks):
        tracker.obj_detect = obj_detect
        tracker.reid_network = reid_network
        tracker.reset()

        # 初始化tracker的中央凹优化部分
        print(f'img_width: {seq.im_width}, img_height: {seq.im_height}')
        if tracker.fovea_switch:
            tracker.init_fovea_optimizer(img_width=seq.im_width, img_height=seq.im_height)

        _log.info(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)

        results = {}
        if load_results:
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()

            for frame_data in tqdm(seq_loader):
                with torch.no_grad():
                    tracker.step(frame_data)

            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if interpolate:
                results = interpolate_tracks(results)

            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

        
        debug_output_dir = osp.join(output_dir, f"{seq}")
        if not os.path.exists(debug_output_dir):
            os.makedirs(debug_output_dir)

        # 0902调试修改
        # 将tracker记录的_raw_detections信息写入detections.txt中
        with open(os.path.join(debug_output_dir, "detections.txt"), "w") as detection_file:
            for frame_id, raw_detections in tracker._raw_detections.items():
                # print(f"raw detections is {raw_detections} at frame {frame_id}")
                # 遍历该帧的所有检测结果
                raw_detection_fovea_mask = tracker._raw_detections_fovea_mask[frame_id]
                for _index, single_raw_detection_score in enumerate(tracker._raw_detections_scores[frame_id]):
                    single_raw_bbox = raw_detections[_index]
                    single_raw_score = single_raw_detection_score
                    single_raw_fovea_mask = raw_detection_fovea_mask[_index]
                    detection_file.write(f"{frame_id + 1} {single_raw_bbox[0]} {single_raw_bbox[1]} {single_raw_bbox[2]} {single_raw_bbox[3]} {single_raw_score} {single_raw_fovea_mask}\n")
                
        # 0903调试修改
        # 写入每一帧中央凹区域的位置
        # if tracker.fovea_switch:
        #     with open(os.path.join(output_dir, f"{seq}", "fovea_positions.txt"), "w") as fovea_pos_file:
        #         for frame_id, fovea_pos in tracker._fovea_positions.items():
        #             fovea_pos_file.write(f"{frame_id + 1} {fovea_pos[0]} {fovea_pos[1]} {fovea_pos[2]} {fovea_pos[3]}\n")
            
        #     # 将截取的中央凹区域图像写入到指定目录下
        #     fovea_raw_img_output_path = os.path.join(output_dir, f"{seq}", "fovea_raw_imgs")
        #     if not os.path.exists(fovea_raw_img_output_path):
        #         os.makedirs(fovea_raw_img_output_path)
            
        #     # 先清空之前生成的图片
        #     for file in os.listdir(fovea_raw_img_output_path):
        #         os.remove(os.path.join(fovea_raw_img_output_path, file))    
            
        #     for frame_id, fovea_raw_img in tracker._raw_fovea_images.items():
        #         fovea_raw_img = np.transpose(fovea_raw_img, (1, 2, 0))
        #         fovea_raw_img = (fovea_raw_img * 255).clip(0, 255)
        #         fovea_raw_img = fovea_raw_img.astype(np.uint8)
        #         fovea_raw_img_pil = Image.fromarray(fovea_raw_img)
        #         fovea_raw_img_pil.save(os.path.join(fovea_raw_img_output_path, f"{(frame_id + 1):06d}.jpg"))

        if seq.no_gt:
            _log.info("No GT data for evaluation available.")
        else:
            accum = get_mot_accum(results, seq_loader)
            fp_events = {}
            fp_bboxes = {}

            match_events = {}
            match_boxes = {}
    
            # 获取所有事件的DataFrame
            events_df = accum.events
            
            # 筛选出所有标记为FP的行
            fp_rows = events_df[events_df.Type == 'FP']
            # 筛选出所有标记为match的行
            match_rows = events_df[events_df.Type == 'MATCH']
            
            # 0903调试修改
            # 记录所有id switch事件
            
            # 遍历筛选后的行，构建结果字典
            for index, row in fp_rows.iterrows():
                frame_id = row.name[0]  # 帧编号
                hyp_id = row.HId  # 锚框ID
                # if frame_id % 50 == 0:
                #     print(f"frame_id: {frame_id}, hyp_id: {hyp_id}, raw row {row}")
                
                # 如果帧编号不在字典中，则初始化为空列表
                if frame_id not in fp_events:
                    fp_events[frame_id] = []
                    fp_bboxes[frame_id] = []
                
                # 将锚框ID添加到对应帧编号的列表中
                fp_events[frame_id].append(hyp_id)
                # 将对应的锚框值添加到对应帧编号的锚框列表中
                # 根据代码，hyp_id就是轨迹ID，mot_accum计算时，逐帧遍历，
                # 为每一帧的预测值和实际值做匹配，预测数据为(track_id, x1, y1, w, h, score)
                fp_bboxes[frame_id].append(results[hyp_id][frame_id][:4])

            for index, row in match_rows.iterrows():
                frame_id = row.name[0]
                hyp_id = row.HId
                if frame_id not in match_events:
                    match_events[frame_id] = []
                    match_boxes[frame_id] = []
                match_events[frame_id].append(hyp_id)
                match_boxes[frame_id].append(results[hyp_id][frame_id][:4])
            
            

            # print(f"Root dir for output fp files: {output_fp_dir}")
            
            with open(osp.join(debug_output_dir, "fp.txt"), "w") as fp_file:

                # 遍历字典，输出每一帧的FP个数
                for frame_id, fp_list in fp_events.items():
                    # if frame_id % 50 == 0:
                    #     print(f"frame_id: {frame_id}, fp_count: {len(fp_list)}")
                        # print(f"bboxes: {fp_bboxes[frame_id]}")

                    for bbox in fp_bboxes[frame_id]:
                        fp_file.write(f"{frame_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

            
            with open(osp.join(debug_output_dir, "match.txt"), "w") as match_file:
                for frame_id, match_list in match_events.items():
                    # if frame_id % 50 == 0:
                    #     print(f"frame_id: {frame_id}, match_count: {len(match_list)}")
                        # print(f"bboxes: {match_boxes[frame_id]}")
                    for index, bbox in enumerate(match_boxes[frame_id]):
                        match_file.write(f"{frame_id} {match_list[index]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
            
            mot_accums.append(accum)

        if write_images:
            plot_sequence(
                results,
                seq,
                osp.join(output_dir, str(dataset), str(seq)),
                write_images)

    if time_total:
        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        _log.info("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt],
                            generate_overall=True)
