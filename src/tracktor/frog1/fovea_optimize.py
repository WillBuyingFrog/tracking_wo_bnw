import torch
import argparse
import random
import math
import os
import cv2
import matplotlib.pyplot as plt


OVERLAP_TRICK = True

class Logger:
    def __init__(self, file_path):
        """初始化日志记录器，指定日志输出的文件位置并创建文件"""
        self.file_path = file_path
        # 打开文件，准备写入，如果文件不存在则创建，如果文件已存在则清空已有内容
        self.file = open(file_path, 'w')

    def write_log(self, message):
        """向指定的文件中写入日志信息"""
        self.file.write(message + '\n')

    def close(self):
        """保存并关闭日志文件"""
        self.file.close()


# 将单次模拟退火的结果画成图
def visualize_ann_result(tlwhs, fovea_x, fovea_y, fovea_width, fovea_height, img_width, img_height,
                         save_path='results/', img_name='test.jpg'):
    

    # 画出目标框
    for tlwh in tlwhs:
        x, y, w, h = tlwh
        plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], 'r')
    # 中央凹区域的左上角坐标为(fovea_x,fovea_y)，右下角坐标为(fovea_x+fovea_width, fovea_y+fovea_height)
    plt.plot([fovea_x, fovea_x + fovea_width, fovea_x + fovea_width, fovea_x, fovea_x],
             [fovea_y, fovea_y, fovea_y + fovea_height, fovea_y + fovea_height, fovea_y], 'b')

    # 设置当前画的图的横轴画到img_width，纵轴最大值画到img_height
    plt.xlim(0, img_width)
    plt.ylim(0, img_height)

    # 保存图像到save_path/img_name
    plt.savefig(os.path.join(save_path, img_name))
    plt.close()


def visualize_all_results(roi_tlwhs, target_tlwhs, fovea_tlwh, roi_color, target_color, fovea_color,
                          mode=1, img=None, img_width=-1, img_height=-1,
                          save_path=None, img_name='test.jpg'):
    # 需要在img上可视化锚框
    if mode == 2:
        img = img.copy()
        if roi_tlwhs is not None:
            for tlwh in roi_tlwhs:
                x, y, w, h = tlwh
                img = cv2.rectangle(img, (x, y), (x + w, y + h), roi_color, 2)
        if target_tlwhs is not None:
            for tlwh in target_tlwhs:
                x, y, w, h = tlwh
                # convert xywh to integer
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), target_color, 2)
        if fovea_tlwh is not None:
            x, y, w, h = fovea_tlwh
            img = cv2.rectangle(img, (x, y), (x + w, y + h), fovea_color, 2)
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, img_name), img)
        return img
    elif mode == 1:
        # 需要在img_shape给出的最大横纵坐标的坐标系中，可视化锚框和中央凹区域
        # 用matplotlib
        if roi_tlwhs is not None:
            for roi_tlwh in roi_tlwhs:
                x, y, w, h = roi_tlwh
                plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], roi_color)
        if target_tlwhs is not None:
            for target_tlwh in target_tlwhs:
                x, y, w, h = target_tlwh
                plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], target_color)
        if fovea_tlwh is not None:
            x, y, w, h = fovea_tlwh
            plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], fovea_color)
        plt.xlim(0, img_width)
        plt.ylim(0, img_height)
        plt.savefig(os.path.join(save_path, img_name))


def calculate_overlap_area(x, y, tlwh, fovea_width, fovea_height):
    # 中央凹区域的左上角坐标
    fovea_left = x
    fovea_top = y

    # 中央凹区域的右下角坐标
    fovea_right = fovea_left + fovea_width
    fovea_bottom = fovea_top + fovea_height

    # 目标的左上角和右下角坐标
    target_left = tlwh[0]
    target_top = tlwh[1]
    target_right = target_left + tlwh[2]
    target_bottom = target_top + tlwh[3]

    # 计算重叠区域的坐标
    overlap_left = max(fovea_left, target_left)
    overlap_top = max(fovea_top, target_top)
    overlap_right = min(fovea_right, target_right)
    overlap_bottom = min(fovea_bottom, target_bottom)

    # 计算重叠区域的宽度和高度
    overlap_width = max(overlap_right - overlap_left, 0.0)
    overlap_height = max(overlap_bottom - overlap_top, 0.0)

    # 计算重叠面积
    overlap_area = overlap_width * overlap_height
    return overlap_area


def calculate_coverage(x, y, tlwh, fovea_width, fovea_height):
    overlap_area = calculate_overlap_area(x, y, tlwh, fovea_width, fovea_height)
    target_area = tlwh[2] * tlwh[3]
    coverage = overlap_area / target_area

    if OVERLAP_TRICK:
        # 鉴于目标追踪器抽取特征的特性，能完整覆盖到整个目标比最大化覆盖率之和更重要，
        # 应当在确保完整覆盖的情况下最大化覆盖率。
        # 原始算法仅将所有覆盖率相加，不能体现确保完整覆盖目标的重要性，容易出现每个目标“雨露均沾”却都没有完整包围目标的情况
        # 这种情况下，目标追踪器抽取特征不稳定（一部分清晰一部分不清晰），提升效果或许有限
        # 所以当OVERLAP_TRICK开启时，本函数返回的coverage值不再表示原始的覆盖率，而是表示更广义上的描述中央凹区域覆盖本目标的“得分”
        # 对覆盖率不足1的情况施加惩罚，使得原始覆盖率不足1时，“得分”迅速下降
        coverage = coverage ** 7

    return coverage

def total_coverage(x, y, tlwhs, fovea_width, fovea_height,
                   opt_version=1, box_weight=[]):
    total = 0
    for index, tlwh in enumerate(tlwhs):
        if opt_version == 2:
            total += (calculate_coverage(x, y, tlwh, fovea_width, fovea_height) * box_weight[index])
        elif opt_version == 1:
            total += calculate_coverage(x, y, tlwh, fovea_width, fovea_height)
    return total

def simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height,
                         init_x=None, init_y=None,
                         opt_version=1, box_weight=[],
                         visualize_path='../results/', visualize=False, marker='mark'):
    if visualize:
        logger = Logger(os.path.join(visualize_path, f'annealing_result_{marker}.txt'))
    

    # 计算所有目标框的中心点
    center_x, center_y = 0.0, 0.0
    for tlwh in tlwhs:
        _x = tlwh[0] + tlwh[2] / 2
        _y = tlwh[1] + tlwh[3] / 2
        center_x += _x
        center_y += _y
    center_x /= len(tlwhs)
    center_y /= len(tlwhs)

    # 定义初始状态
    if init_x is not None:
        current_x = init_x
    else:
        current_x = max(1, center_x - fovea_width)

    if init_y is not None:
        current_y = init_y
    else:
        current_y = max(1, center_y - fovea_height)
    current_score = total_coverage(current_x, current_y, tlwhs, fovea_width, fovea_height)
    if visualize:
        logger.write_log(f'New round. Initial position x={current_x:.2f}, y={current_y:.2f}, score={current_score:.2f}')

    # 温度调度参数
    initial_temp = 50.0
    final_temp = 0.05
    alpha = 0.9
    temp = initial_temp

    # 记录循环轮次
    counter = 0

    # 模拟退火迭代
    while temp > final_temp:
        counter += 1

        # 随机选择新的状态（邻域函数）
        if counter > 1:
            next_x = current_x + random.uniform(-75, 75)  # 后续需要调整步长以适应不同宽高的图片
            next_y = current_y + random.uniform(-75, 75)

            # 不断随机采样，直到采样得到新的合法坐标
            new_pos_eff = 75.0
            while next_x + fovea_width > img_width or next_x <= 0:
                new_pos_eff *= 1.2
                new_pos_eff = min(new_pos_eff, fovea_width)
                next_x = current_x + random.uniform(-new_pos_eff, new_pos_eff) 
            new_pos_eff = 10.0
            while next_y + fovea_height > img_height or next_y <= 0:
                new_pos_eff *= 1.2
                new_pos_eff = min(new_pos_eff, fovea_height)
                next_y = current_y + random.uniform(-new_pos_eff, new_pos_eff)
        else:
            next_x, next_y = current_x, current_y
            # visualize_ann_result(tlwhs, next_x, next_y, fovea_width, fovea_height,
            #                       img_width, img_height,
            #                       save_path=visualize_path, img_name=f'ann_{marker}_initial.jpg')


        next_score = total_coverage(next_x, next_y, tlwhs, fovea_width, fovea_height,
                                    opt_version=opt_version, box_weight=box_weight)

        # 计算接受概率
        accept_probability = math.exp((next_score - current_score) / temp)
        if visualize:
            logger.write_log(f'\tRound {counter} - Temp: {temp:.2f}, Current score: {current_score:.2f}, Next score: {next_score:.2f}, Accept probability: {accept_probability:.2f}')

        if next_score > current_score or random.random() < accept_probability:
            if visualize:
                logger.write_log(f'\t\tAccept new position: x={next_x:.2f}, y={next_y:.2f}, score={next_score:.2f}')
            current_x, current_y = next_x, next_y
            current_score = next_score

        # 降低温度
        temp *= alpha
    if visualize:
        logger.write_log(f'\tFinal position: x={current_x:.2f}, y={current_y:.2f}, score={current_score:.2f}')
        logger.close()  

    return current_x, current_y, current_score

def optimize(tlwhs, fovea_width, fovea_height, img_width, img_height,
             init_x=None, init_y=None, epochs=1, algo='annealing',
             opt_version=1, box_weight=[],
             visualize=False, visualize_path='../results',
             dry_run=False):

    if dry_run:
        return img_width // 4, img_height // 4

    if algo == 'annealing':
        avg_x, avg_y, total_score = 0, 0, 0.0
        results = []
        for i in range(epochs):
            x, y, current_score = simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height,
                                       init_x=init_x, init_y=init_y,
                                       opt_version=opt_version, box_weight=box_weight,
                                       visualize=visualize, visualize_path=visualize_path)
            results.append((x, y, current_score))
            if visualize:
                visualize_ann_result(tlwhs, x, y, fovea_width, fovea_height, img_width, img_height,
                                      save_path=visualize_path, img_name=f'test_{i + 1}.jpg')
        
        # 按照每个轮次返回的score进行加权平均
        for i in range(epochs):
            x, y, score = results[i]
            avg_x += x * score
            avg_y += y * score
            total_score += score
            # print(f'Epoch {i+1} - Simulated annealing result x={x}, y={y}, score={score:.2f}')
            # print(f'      avg_x={avg_x:.2f}, avg_y={avg_y:.2f}, total_score={total_score:.2f}')

        if total_score < 0.1:
            # return the center of the image
            avg_x, avg_y = img_width / 2, img_height / 2
            return avg_x, avg_y
        
        avg_x /= (total_score)
        avg_y /= (total_score)

        # print(f'Average position: x={avg_x:.2f}, y={avg_y:.2f}')
        if visualize:
            visualize_ann_result(tlwhs, avg_x, avg_y, fovea_width, fovea_height,
                                  img_width, img_height, save_path=visualize_path, img_name=f'test_avg.jpg',)
        
        return avg_x, avg_y
    else:
        return -1, -1


if __name__ == '__main__':

    # 创建一个argument parser
    parser = argparse.ArgumentParser(description='中央凹区域位置优化算法')
    parser.add_argument('--algo_type', default='annealing', type=str, help='优化算法类型')
    parser.add_argument('--img_width', default=800, type=int, help='图像宽度')
    parser.add_argument('--img_height', default=600, type=int, help='图像高度')
    parser.add_argument('--fovea_width', default=400, type=int, help='中央凹区域宽度')
    parser.add_argument('--fovea_height', default=300, type=int, help='中央凹区域高度')
    parser.add_argument('--exp_epochs', default=1, type=int, help='算法迭代次数')
    parser.add_argument('--clear_previous', default=True, type=bool, help='是否清除之前的可视化结果')
    parser.add_argument('--testdata_path', default='no', type=str)

    args = parser.parse_args()
    img_width = args.img_width
    img_height = args.img_height
    fovea_width = args.fovea_width
    fovea_height = args.fovea_height
    exp_epochs = args.exp_epochs
    clear_previous = args.clear_previous
    testdata_path = args.testdata_path

    if clear_previous:
        if os.path.exists('results'):
            # 删除其中所有的.jpg文件
            for file in os.listdir('results'):
                if file.endswith('.jpg'):
                    os.remove(os.path.join('results', file))



    
    # 示例: 假设有一个目标列表
    tlwhs = [
        torch.tensor([80, 100, 35, 100]),
        torch.tensor([100, 120, 35, 100]),
        torch.tensor([120, 150, 35, 100]),
        torch.tensor([150, 180, 35, 100]),
        torch.tensor([190, 210, 35, 100]), 
        torch.tensor([500, 100, 35, 100]), 
        torch.tensor([550, 100, 50, 150]), 
        # ... 更多目标
    ]

    
    if testdata_path != 'no':
        print(f'Loading tlwhs data from file: {testdata_path}')
        # the tlwhs data in the testdata_path file is organized as follows:
        # four float numbers each row, each row represents a target's tlwh
        # top_x left_y width height
        # for example, a testdata_path file that contains two targets should look like:
        # 10.0 20.0 30.0 50.0
        # 20.0 30.0 30.0 50.0
        # now read the file and load the tlwhs data
        tlwhs = []
        with open(testdata_path, 'r') as f:
            for line in f:
                tlwhs.append(torch.tensor([float(x) for x in line.strip().split()]))
        print(f'Loaded tlwhs: {tlwhs}')

    if args.algo_type == 'grad':
        # 中央凹区域的位置
        x = 100
        y = 100

        # 计算总覆盖率
        total_cov = total_coverage(x, y, tlwhs, 100, 100)

        # 反向传播
        total_cov.backward()

        # 查看梯度
        print(x.grad, y.grad)
    elif args.algo_type == 'annealing':
        
        avg_x, avg_y, total_score = 0, 0, 0.0
        results = []
        for i in range(args.exp_epochs):
            x, y, current_score = simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height,
                                       init_x=None, init_y=None,
                                       visualize=True, visualize_path='results/', marker=i+1)
            results.append((x, y, current_score))
            # print(f'Epoch {i+1} - Simulated annealing result x={x}, y={y}')
            if (i + 1) % 5 == 0:
                visualize_ann_result(tlwhs, x, y, fovea_width, fovea_height, img_width, img_height,
                                      save_path='results/', img_name=f'test_{i + 1}.jpg',)

        
        # 按照每个轮次返回的score进行加权平均
        for i in range(args.exp_epochs):
            x, y, score = results[i]
            avg_x += x * score
            avg_y += y * score
            total_score += score
            print(f'Epoch {i+1} - Simulated annealing result x={x}, y={y}, score={score:.2f}')
            print(f'      avg_x={avg_x:.2f}, avg_y={avg_y:.2f}, total_score={total_score:.2f}')
        avg_x /= (total_score)
        avg_y /= (total_score)

        print(f'Average position: x={avg_x:.2f}, y={avg_y:.2f}')
        visualize_ann_result(tlwhs, avg_x, avg_y, fovea_width, fovea_height, img_width, img_height, save_path='results/', img_name=f'test_avg.jpg',)

        
        # print(f'Simulated annealing result x={x}, y={y}')
