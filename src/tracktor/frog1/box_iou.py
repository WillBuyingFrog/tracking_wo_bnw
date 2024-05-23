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