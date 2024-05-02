import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

import argparse
from collections import OrderedDict
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os


class Fovea_FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, device=None):
        
        backbone = resnet_fpn_backbone('resnet50', False)
        super(Fovea_FRCNN_FPN, self).__init__(backbone, num_classes, box_detections_per_img=300)
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None
        self.device = device

    def detect(self, images):

        # if self.device is not None:
        #     print(f'Image tensor type {type(images)}')
        #     images = images.to(self.device)
        #     print(f'Using device {self.device}')
        #     print(f'Image tensor type {type(images)}')
            
        detections = self(images)[0]

        return detections['boxes'].detach(), detections['scores'].detach(), detections['labels'].detach()
    
    def load_image(self, images):
        if self.device is not None:
            images = images.to(self.device)
        elif torch.cuda.is_available():
            images = images.cuda()

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])


# 实现函数：
#   输入：检测到的中央凹部分目标原始boxes，中央凹区域在原始图像中的tlwh，原图长宽缩小比例（默认长宽缩小比例一致）
#   输出：中央凹区域图像中检测到的所有目标的boxes，
#       要求所有boxes均已经转换到缩小后原图的坐标位置


def get_processed_boxes(fovea_boxes, fovea_pos, compress_ratio=[3.0, 3.0]):

    fovea_boxes = fovea_boxes.cpu()

    processed_boxes = []

    for fovea_box in fovea_boxes:
        
        processed_box = np.zeros_like(fovea_box)
        processed_box[0] = (fovea_pos[0] + fovea_box[0]) / compress_ratio[0]
        processed_box[1] = (fovea_pos[1] + fovea_box[1]) / compress_ratio[1]
        processed_box[2] = (fovea_pos[0] + fovea_box[2]) / compress_ratio[0]
        processed_box[3] = (fovea_pos[1] + fovea_box[3]) / compress_ratio[1]
        processed_boxes.append(processed_box)
    
    # 将origin_boxes转成pytorch tensor
    processed_boxes = torch.tensor(processed_boxes, dtype=torch.float32)

    # 将Processed_boxes挪到cuda上
    if torch.cuda.is_available():
        processed_boxes = processed_boxes.cuda()
    
    return processed_boxes
