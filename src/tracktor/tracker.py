from collections import deque

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchreid import metrics
from torchvision.ops.boxes import clip_boxes_to_image, nms
from torchvision.transforms import ToTensor, ToPILImage

from .utils import (bbox_overlaps, get_center, get_height, get_width, make_pos,
                    warp_pos)


from .frog1.fovea_obj_detect import get_processed_boxes
from .frog1.fovea_optimize import FoveaOptimizer
from .frog1.box_iou import box_iou, check_bbox_in_fovea_region
from .frog1.frog_logger import FrogLogger

import os
from PIL import Image
from PIL import ImageDraw


class Tracker:
    """The main tracking file, here is where magic happens."""
    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg, 
                 origin_obj_detect=None):
        self.obj_detect = obj_detect
        self.reid_network = reid_network
        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']

        self.warp_mode = getattr(cv2, tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']

        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        # TODO 增加中央凹区域相关参数
        # 包括：检测中央凹区域的目标检测器
        #       中央凹区域长宽



        # 在下采样的实验模式下，检测原图指定区域目标的模型
        self.fovea_switch = tracker_cfg['frog_fovea']
        self.origin_obj_detect = origin_obj_detect
        self.compress_ratio = tracker_cfg['frog_compress_ratio']
        self.fovea_optimizer = None
        self.fovea_scale = tracker_cfg['frog_fovea_scale']
        self.do_fovea_logs = tracker_cfg['frog_logger']
        self.fovea_pos_in_original_image = (-1, -1, -1, -1)
        if self.do_fovea_logs:
            __path =  tracker_cfg['frog_logger_path']
            self.frog_logger_path = f'{__path}/fovea_logger.log'
            self.fovea_logger = FrogLogger(self.frog_logger_path)


    def init_fovea_optimizer(self, img_width, img_height):
        fovea_width = int(img_width * self.fovea_scale)
        fovea_height = int(img_height * self.fovea_scale)
        # print(f'fovea_width: {fovea_width}, fovea_height: {fovea_height}')
        self.fovea_optimizer = FoveaOptimizer(img_width=img_width, img_height=img_height, init_image_path=None,
                                              region_scale=0.025, pixel_change_threshold=70,
                                              fovea_width=fovea_width, fovea_height=fovea_height,
                                              is_PIL=True)


    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                self.track_num + i,
                new_det_features[i].view(1, -1),
                self.inactive_patience,
                self.max_features_num,
                self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
            ))
        self.track_num += num_new

    def regress_tracks(self, blob, _fovea_img=None, compressed_fovea_pos=None):
        """Regress the position of the tracks and also checks their scores."""
        pos = self.get_pos()

        boxes = []
        scores = []

        pos_inside_fovea = []
        pos_inside_fovea_ind = []
        pos_outside_fovea = []
        pos_outside_fovea_ind = []
        _counter = -1

        # origin_fovea_pos =  self.fovea_pos_in_original_image

        if _fovea_img is not None and self.fovea_switch:
            _pos = pos.clone().cpu().tolist()
            # self.fovea_logger.write_log(f"\tFovea region in original image is {self.fovea_pos_in_original_image}, in compressed image is {compressed_fovea_pos}")
            for single_pos in _pos:
                _counter += 1
                if check_bbox_in_fovea_region(single_pos, compressed_fovea_pos):
                    # if self.do_fovea_logs:
                        # self.fovea_logger.write_log(f"\tOld box {single_pos} is in fovea region, ind{_counter}")
                    # 计算得到该锚框在中央凹区域下的坐标
                    pos_inside_fovea.append(single_pos)
                    pos_inside_fovea_ind.append(_counter)
                else:
                    # if self.do_fovea_logs:
                        # self.fovea_logger.write_log(f"\tOld box {single_pos} is outside fovea region, ind{_counter}")
                    pos_outside_fovea.append(single_pos)
                    pos_outside_fovea_ind.append(_counter)
        
        # print(f"Position outside fovea: {pos_inside_fovea} with indices: {pos_outside_fovea_ind}")

        # print(f"Position inside fovea: {pos_inside_fovea} with indices: {pos_inside_fovea_ind}")

        # 兼容已有实现，需要把pos_inside_fovea和pos_outside_fovea转换为tensor
        pos_inside_fovea = torch.Tensor(pos_inside_fovea)
        pos_outside_fovea = torch.Tensor(pos_outside_fovea)

        if len(pos_inside_fovea):
            # 先把pos_inside_fovea的坐标转换到高清小区域图上
            origin_fovea_x, origin_fovea_y, origin_fovea_w, origin_fovea_h = self.fovea_pos_in_original_image
            _pos_inside_fovea = pos_inside_fovea.clone()
            _pos_inside_fovea[:, 0], _pos_inside_fovea[:, 2] = _pos_inside_fovea[:, 0] * self.compress_ratio[0] - origin_fovea_x, _pos_inside_fovea[:, 2] * self.compress_ratio[0] - origin_fovea_x
            _pos_inside_fovea[:, 1], _pos_inside_fovea[:, 3] = _pos_inside_fovea[:, 1] * self.compress_ratio[1] - origin_fovea_y, _pos_inside_fovea[:, 3] * self.compress_ratio[1] - origin_fovea_y
            # 再执行检测
            _boxes_inside, _scores_inside = self.origin_obj_detect.predict_boxes(_pos_inside_fovea)
            # 然后返回的boxes还要再变换回低清大区域图上
            _boxes_inside = get_processed_boxes(_boxes_inside, self.fovea_pos_in_original_image, self.compress_ratio)

            # TEMP_OUTPUT_PATH = '/home/user/frog/mot-dbt/tracking_wo_bnw/output/frog_log_0830'
            # if self.do_fovea_logs:
            #     # 把blob里的图片复制过来，然后用cv2画上_boxes_inside，再保存到TEMP_OUTPUT_PATH里
            #     _img = cv2.imread(blob['img_path'][0])
            #     for box in _boxes_inside:
            #         _box = box.clone().cpu().numpy()
            #         x0, y0, x1, y1 = _box
            #         x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            #         print(f"x0, y0, x1, y1: {x0}, {y0}, {x1}, {y1}")
            #         cv2.rectangle(_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #     cv2.imwrite(f'{TEMP_OUTPUT_PATH}/frame_{self.im_index}_inside.jpg', _img)

        
        if len(pos_outside_fovea):
            # print(f"Position outside fovea: {pos_outside_fovea} with indices: {pos_outside_fovea_ind}")
            _boxes_outside, _scores_outside = self.obj_detect.predict_boxes(pos_outside_fovea)
        
        for i in range(len(self.tracks) - 1, -1, -1):
            # 用n^2复杂度方法找每一条track的检测结果
            # 先找在中央凹区域内的
            found = False
            if len(pos_inside_fovea_ind):
                for j in range(len(pos_inside_fovea_ind)):
                    if i == pos_inside_fovea_ind[j]:
                        boxes.append(_boxes_inside[j])
                        scores.append(_scores_inside[j])
                        found = True
                        break
            # 再找在中央凹区域外的
            if len(pos_outside_fovea_ind):
                for j in range(len(pos_outside_fovea_ind)):
                    if i == pos_outside_fovea_ind[j]:
                        boxes.append(_boxes_outside[j])
                        scores.append(_scores_outside[j])
                        found = True
                        break
            
            assert found == True, f'No detection found for track {i}'

        boxes = torch.stack(boxes, dim=0)
        scores = torch.stack(scores, dim=0)
        
        # _img = cv2.imread(blob['img_path'][0])
        # for box in boxes:
        #     _box = box.clone().cpu().numpy()
        #     x0, y0, x1, y1 = _box
        #     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        #     cv2.rectangle(_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # cv2.imwrite(f'./output/frog_log_0830/frame_{self.im_index}_all.jpg', _img)

        # if self.do_fovea_logs:
        #     self.fovea_logger.write_log(f"\tIn regress_tracks, optimized boxes and scores:")
        #     for i in range(len(self.tracks)-1, -1, -1):
        #         self.fovea_logger.write_log(f"\t\t{i} {boxes[i]}, {scores[i]}")
        #     _boxes, _scores = self.obj_detect.predict_boxes(pos)
        #     self.fovea_logger.write_log(f"\tIn regress_tracks, boxes and scores without optimization:")
        #     for i in range(len(self.tracks)-1, -1, -1):
        #         self.fovea_logger.write_log(f"\t\t{i} {_boxes[i]}, {_scores[i]}")


        # regress
        # boxes, scores = self.obj_detect.predict_boxes(pos)
        pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)

        return torch.Tensor(s[::-1]).cuda()

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with new detections."""
        new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            new_det_features = self.get_appearances(blob, new_det_pos)

            if len(self.inactive_tracks) >= 1:
                # calculate appearance distances
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
                                               for feat in new_det_features], dim=1))
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # calculate IoU distances
                if self.reid_iou_threshold:
                    iou = bbox_overlaps(pos, new_det_pos)
                    iou_mask = torch.ge(iou, self.reid_iou_threshold)
                    iou_neg_mask = ~iou_mask
                    # make all impossible assignments to the same add big value
                    dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def get_appearances(self, blob, pos):
        """Uses the siamese CNN to get the features for all active tracks."""
        crops = []
        for r in pos:
            x0 = int(r[0])
            y0 = int(r[1])
            x1 = int(r[2])
            y1 = int(r[3])
            if x0 == x1:
                if x0 != 0:
                    x0 -= 1
                else:
                    x1 += 1
            if y0 == y1:
                if y0 != 0:
                    y0 -= 1
                else:
                    y1 += 1
            crop = blob['img'][0, :, y0:y1, x0:x1].permute(1, 2, 0)
            crops.append(crop.mul(255).numpy().astype(np.uint8))

        new_features = self.reid_network(crops)

        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
            cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            warp_matrix = torch.from_numpy(warp_matrix)

            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)
                # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg['enabled']:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg['center_only']:
                vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = torch.stack(vs).mean(dim=0)
            self.motion_step(t)

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)

    def step(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())

        # 加载原图
        if self.fovea_switch:
            origin_img_path = blob['origin_img_path'][0]
            if os.path.exists(origin_img_path):
                origin_img = Image.open(origin_img_path).convert("RGB")
                origin_img = ToTensor()(origin_img)
            else:
                print(f'origin_img_path: {origin_img_path} not exists')
        
        if self.do_fovea_logs:
            self.fovea_logger.write_log(f"Frame {self.im_index} start")

        ###########################
        # Look for new detections #
        ###########################

        self.obj_detect.load_image(blob['img'])

        if self.public_detections:
            dets = blob['dets'].squeeze(dim=0)
            if dets.nelement() > 0:
                boxes, scores = self.obj_detect.predict_boxes(dets)
            else:
                boxes = scores = torch.zeros(0).cuda()
        else:
            boxes, scores = self.obj_detect.detect(blob['img'])
            if self.fovea_switch:
                    # TODO 实现基于模拟退火优化中央凹区域的效果
                    # 目前先默认中央凹区域永远是最中间的部分
                    img_h, img_w = origin_img.size(1), origin_img.size(2)
                    fovea_x, fovea_y = img_w // 4, img_h // 4
                    fovea_w, fovea_h = img_w // 2, img_h // 2
                    # 将blob['img'][0]转成cv2格式
                    # cv2_img = blob['img'][0].permute(1, 2, 0).mul(255).byte().cpu().numpy()

                    prev_online_boxes = []

                    if len(self.tracks):
                        for track in self.tracks:
                            track_pos = track.pos[0].cpu().numpy()
                            prev_online_boxes.append(track_pos)

                    fovea_x, fovea_y = self.fovea_optimizer.get_fovea_position(current_frame_img=blob['img'][0],
                                                            prev_online_boxes=prev_online_boxes,
                                                            visualize=False, visualize_mark=f'{self.im_index}', visualize_path='./debug/')
                    # print('Fovea optimize complete')
                    # 如果fovea_x, fovea_y is NaN, 则使用默认值
                    if np.isnan(fovea_x) or np.isnan(fovea_y):
                        fovea_x, fovea_y = int(img_w / 4 / self.compress_ratio[0]), int(img_h / 4 / self.compress_ratio[1])
                    # 放大回原图尺寸
                    compressed_fovea_pos = (fovea_x, fovea_y, fovea_w / self.compress_ratio[0], fovea_h / self.compress_ratio[1])
                    # print(f'[Before]fovea_x: {fovea_x}, fovea_y: {fovea_y}')
                    fovea_x, fovea_y = int(fovea_x * self.compress_ratio[0]), int(fovea_y * self.compress_ratio[1])

                    
                    fovea_pos = (fovea_x, fovea_y, fovea_w, fovea_h)
                    self.fovea_pos_in_original_image = fovea_pos
                    # print(f'fovea_pos: {fovea_pos}')
                    
                    # 截取出给定中央凹区域tlwh对应的原始图像内容
                    # origin_img 是一个pytorch的tensor张量
                    fovea_img = origin_img[:, fovea_y:fovea_y+fovea_h, fovea_x:fovea_x+fovea_w]
                    _fovea_img = torch.stack([fovea_img], dim=0)
                    # print(f'origin_img shape: {origin_img.shape}, fovea_img shape: {fovea_img.shape}')

                    # 送入中央凹区域的目标检测器进行检测，获得该目标检测器输出的boxes和scores
                    # fovea_boxes, fovea_scores = self.origin_obj_detect.detect(_fovea_img)
                    self.origin_obj_detect.load_image(_fovea_img)
                    fovea_boxes, fovea_scores = self.obj_detect.detect(_fovea_img)

                    # 将所有中央凹区域得到的目标boxes转换到blob['img']上
                    processed_fovea_boxes = get_processed_boxes(fovea_boxes=fovea_boxes, fovea_pos=fovea_pos,
                                                    compress_ratio=self.compress_ratio)
                    
                    # if self.do_fovea_logs and self.im_index % 10 == 0:
                    #     # 打印本帧中全视角图像中检测出的锚框
                    #     self.fovea_logger.write_log(f'{len(boxes)} boxes in wide FOV image')
                    #     for box in boxes:
                    #         self.fovea_logger.write_log(f'\t {box}')
                    #     # 打印本帧计算得到的中央凹区域检测出的锚框，打印已经经过坐标变换的
                    #     self.fovea_logger.write_log(f'{len(processed_fovea_boxes)} boxes in fovea image')
                    #     for box in processed_fovea_boxes:
                    #         self.fovea_logger.write_log(f'\t {box}')
                    

                    # 对每个processed_fovea_boxes，检查其与已有的所有boxes的IoU，如果IoU小于0.5，则将其加入到boxes中
                    # 且将其对应的scores加入到scores中
                    # 初始化一个空列表来存储新的boxes和scores
                    # new_boxes = []
                    # new_scores = []
                    # # 遍历每个processed_fovea_boxes
                    # for box, score in zip(processed_fovea_boxes, fovea_scores):
                    #     # 计算当前box与已有boxes的IoU
                    #     ious = box_iou(box.unsqueeze(0), boxes)
                    #     # 如果IoU小于0.5，则将box和score加入到新列表中
                    #     if (ious < 0.5).all():
                    #         new_boxes.append(box)
                    #         new_scores.append(score)
                    
                    # 0830修改：中央凹区域内的锚框，只选用processed_fovea_boxes
                    new_boxes = []
                    new_scores = []
                    # 遍历所有从低清大区域图中得出的锚框，在中央凹区域中的，就不加入到new_boxes中
                    for box, score in zip(boxes, scores):
                        if not check_bbox_in_fovea_region(box, compressed_fovea_pos):
                            new_boxes.append(box)
                            new_scores.append(score)
                    
                    
                    # if self.do_fovea_logs and self.im_index % 10 == 0:
                    #     # 打印所有需要添加的锚框
                    #     self.fovea_logger.write_log(f'{len(new_boxes)} new valid boxes in fovea image')
                    #     for box in new_boxes:
                    #         self.fovea_logger.write_log(f'\t {box}')
                    # if len(new_boxes):
                    #     new_boxes = torch.stack(new_boxes).cuda()
                    #     new_scores = torch.tensor(new_scores).cuda()
                    #     # 将经过变换的中央凹区域boxes、scores与已有的boxes、scores进行合并
                    #     boxes = torch.cat([boxes, new_boxes], dim=0)
                    #     scores = torch.cat([scores, new_scores], dim=0)
                    if len(new_boxes):
                        new_boxes = torch.stack(new_boxes).cuda()
                        new_scores = torch.tensor(new_scores).cuda()

                        boxes = torch.cat([new_boxes, processed_fovea_boxes], dim=0)
                        scores = torch.cat([new_scores, fovea_scores], dim=0)
                    elif len(processed_fovea_boxes):
                        boxes = processed_fovea_boxes
                        scores = torch.tensor(fovea_scores).cuda()

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = torch.gt(scores, self.detection_person_thresh).nonzero(as_tuple=False).view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()
        
        # if self.do_fovea_logs and self.im_index % 10 == 0:
            
        #     self.fovea_logger.write_log(f'Boxes(clipped) for frame {self.im_index}:')
        #     for box in boxes:
        #         self.fovea_logger.write_log(f'\t {box}')

        #     self.fovea_logger.write_log(f'Final boxes for frame {self.im_index}:')
        #     for box in det_pos:
        #         self.fovea_logger.write_log(f'\t {box}')

        ##################
        # Predict tracks #
        ##################

        if len(self.tracks):
            # align
            if self.do_align:
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # regress
            person_scores = self.regress_tracks(blob, _fovea_img, compressed_fovea_pos)

            if len(self.tracks):
                # create nms input

                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                if keep.nelement() > 0 and self.do_reid:
                        new_features = self.get_appearances(blob, self.get_pos())
                        self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # check with every track in a single run (problem if tracks delete each other)
            for t in self.tracks:
                nms_track_pos = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat(
                    [torch.tensor([2.0]).to(det_scores.device), det_scores])
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

                keep = keep[torch.ge(keep, 1)] - 1

                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # try to reidentify tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

            # add new
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([
                t.pos[0].cpu().numpy(),
                np.array([t.score.cpu()])])
        
        # if self.do_fovea_logs and self.im_index % 10 == 0:
        #     # 记录所有目前激活的track在这一帧的位置信息
        #     self.fovea_logger.write_log(f'Active tracks for frame {self.im_index}:')    
        #     for t in self.tracks:
        #         self.fovea_logger.write_log(f'\t{t.id} {t.pos[0].cpu().numpy()}, {t.score.cpu()}')
        #         self.fovea_logger.write_log(f'compare: {self.results[t.id][self.im_index]}')

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]

    def get_results(self):
        return self.results


class Track(object): 
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = metrics.compute_distance_matrix(features, test_features)
        # dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
