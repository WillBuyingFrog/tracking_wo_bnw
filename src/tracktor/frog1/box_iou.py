import torch


def intersect(box1, box2):
    """
    计算两个边界框集合的交集面积。

    Args:
        box1 (torch.Tensor): 第一个边界框集合，形状为(N, 4)。
        box2 (torch.Tensor): 第二个边界框集合，形状为(M, 4)。

    Returns:
        torch.Tensor: 形状为(N, M)的交集面积矩阵。
    """
    # 计算交集的坐标
    max_xy = torch.min(box1[:, 2:], box2[:, 2:])
    min_xy = torch.max(box1[:, :2], box2[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]


def box_iou(box1, box2):
    """
    计算两个边界框之间的交并比（IoU）。

    Args:
        box1 (torch.Tensor): 第一个边界框，形状为(N, 4)，其中N是框的数量。
        box2 (torch.Tensor): 第二个边界框，形状为(M, 4)，其中M是框的数量。

    Returns:
        torch.Tensor: 形状为(N, M)的IoU矩阵，其中每个元素是box1中的一个框与box2中的一个框的IoU。
    """
    # 计算交集的坐标
    inter_area = intersect(box1, box2)

    # 计算并集的面积
    box1_area = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).view(-1, 1)
    box2_area = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).view(1, -1)
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou



def check_bbox_in_fovea_region(bbox, compressed_fovea_pos):
    """
    检查某个tlbr的bbox是否在tlwh表示下的中央凹区域(低清大区域图坐标)内。

    Args:
        bbox(torch.Tensor): 需要检查的bbox。
        compressed_fovea_pos(torch.Tensor): 中央凹区域的位置,形状为 (4,) ,分别是x, y, w, h。    
    """
    bbox_tlwh = torch.tensor([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])

    # 检查bbox是否在中央凹区域内
    if (bbox_tlwh[0] >= compressed_fovea_pos[0]) and (bbox_tlwh[1] >= compressed_fovea_pos[1]) and \
       (bbox_tlwh[0] + bbox_tlwh[2] <= compressed_fovea_pos[0] + compressed_fovea_pos[2]) and \
       (bbox_tlwh[1] + bbox_tlwh[3] <= compressed_fovea_pos[1] + compressed_fovea_pos[3]):
        return True
    else:
        return False
