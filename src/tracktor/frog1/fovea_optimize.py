import torch
from torchvision import transforms
import argparse
import random
import math
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .calculate_roi import FrogROI

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





class FoveaOptimizer():
    def __init__(self, img_width, img_height, init_image_path, 
                 region_scale, pixel_change_threshold, fovea_width, fovea_height,
                 is_PIL):
        self.img_width = img_width
        self.img_height = img_height
        self.init_image_path = init_image_path

        self.region_scale = region_scale
        self.pixel_change_threshold = pixel_change_threshold
        self.fovea_width = fovea_width
        self.fovea_height = fovea_height
        self.is_PIL = is_PIL

        print(f'FoveaOptimizer: img_width={img_width}, img_height={img_height}, init_image_path={init_image_path}, region_scale={region_scale}, pixel_change_threshold={pixel_change_threshold}, fovea_width={fovea_width}, fovea_height={fovea_height}')
        
        self.roi_calculate = FrogROI(image_width=img_width, image_height=img_height, init_image_path=init_image_path,
                                     region_scale=region_scale, pixel_change_threshold=pixel_change_threshold,
                                     is_PIL=is_PIL)
    
    def blur_image(self, img, blur_radius=5):
        if self.is_PIL:
            transform = transforms.GaussianBlur(kernel_size=(blur_radius, blur_radius), sigma=(0.1, 0.1))
            return transform(img).numpy()
        else:
            return cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)


    def store_engram_image_raw(self, img):
        img = self.blur_image(img)
        self.roi_calculate.store_engram_image_raw(img)

    def box_to_tlwh(self, box):
        return torch.tensor([box[0], box[1], box[2] - box[0], box[3] - box[1]])

    def get_fovea_position(self, current_frame_img, prev_online_boxes,
                           visualize=False, visualize_path='../results', visualize_mark='mark'):
        online_tlwhs = []
        tlwhs_weight = []
        if self.is_PIL:
            blurred_img = self.blur_image(current_frame_img)
        else:
            blurred_img = self.blur_image(current_frame_img).astype(np.float32)
        # 将这一帧和已有的engram进行比较
        if len(self.roi_calculate.engram_images):
            roi_tlwhs = self.roi_calculate.compare_engram(blurred_img)
        else:
            roi_tlwhs = []


        # 记录所有的感兴趣区域
        if len(roi_tlwhs) > 0:
            for roi_tlwh in roi_tlwhs:
                online_tlwhs.append(roi_tlwh)
                tlwhs_weight.append(0.5)
        
        # 将这一帧加到计算比较基准的list里并计算比较基准
        self.roi_calculate.store_engram_images_raw(blurred_img)
        self.roi_calculate.calculate_engram()

        # 将截至上一帧的所有track结果转成tlwh格式
        if len(prev_online_boxes) > 0:
            for box in prev_online_boxes:
                online_tlwhs.append(self.box_to_tlwh(box))
                tlwhs_weight.append(3.0)
        
        if len(online_tlwhs) > 0:
            fovea_x, fovea_y = self.optimize(online_tlwhs, fovea_width=self.fovea_width, fovea_height=self.fovea_height,
                                        img_width=self.img_width, img_height=self.img_height,
                                        epochs=1, algo='annealing',
                                        opt_version=2, box_weight=tlwhs_weight,
                                        visualize=visualize, visualize_path=visualize_path, visualize_mark=visualize_mark, 
                                        dry_run=False)
            return fovea_x, fovea_y
        else:
            # 计算合法中央凹区域xy坐标范围
            valid_x_min, valid_y_min = 0, 0 
            valid_x_max, valid_y_max = max(0, self.img_width - self.fovea_width), max(0, self.img_height - self.fovea_height)
            # 返回合法区域xy的中点
            return int((valid_x_min + valid_x_max) / 2), int((valid_y_min + valid_y_max) / 2)
        
    
    # 将单次模拟退火的结果画成图
    def visualize_ann_result(self, tlwhs, fovea_x, fovea_y, fovea_width, fovea_height, img_width, img_height,
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


    def visualize_all_results(self, roi_tlwhs, target_tlwhs, fovea_tlwh, roi_color, target_color, fovea_color,
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


    def calculate_overlap_area(self, x, y, tlwh, fovea_width, fovea_height):
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


    def calculate_coverage(self, x, y, tlwh, fovea_width, fovea_height):
        overlap_area = self.calculate_overlap_area(x, y, tlwh, fovea_width, fovea_height)
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

    def total_coverage(self, x, y, tlwhs, fovea_width, fovea_height,
                    opt_version=1, box_weight=[]):
        total = 0
        for index, tlwh in enumerate(tlwhs):
            if opt_version == 2:
                total += (self.calculate_coverage(x, y, tlwh, fovea_width, fovea_height) * box_weight[index])
            elif opt_version == 1:
                total += self.calculate_coverage(x, y, tlwh, fovea_width, fovea_height)
        return total

    def simulated_annealing(self, tlwhs, fovea_width, fovea_height, img_width, img_height,
                            init_x=None, init_y=None,
                            opt_version=1, box_weight=[],
                            visualize_path='./debug/', visualize=False, marker='mark'):
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
            current_x = max(0, center_x - fovea_width)

        if init_y is not None:
            current_y = init_y
        else:
            current_y = max(0, center_y - fovea_height)
        current_score = self.total_coverage(current_x, current_y, tlwhs, fovea_width, fovea_height)
        if visualize:
            logger.write_log(f'New round. Initial position x={current_x:.2f}, y={current_y:.2f}, score={current_score:.2f}')
            logger.write_log(f'All tlwhs and correspinding box weight:')
            for index, tlwh in enumerate(tlwhs):
                logger.write_log(f'\ttlwh: {tlwh}, weight: {box_weight[index]}')

        # 算法参数
        initial_temp = 50.0
        final_temp = 0.05
        alpha = 0.9
        temp = initial_temp
        next_pos_range = 0.3

        # 记录循环轮次
        counter = 0

        # 计算迭代过程中坐标的合法范围
        valid_x_min, valid_y_min = 0, 0
        valid_x_max, valid_y_max = img_width - fovea_width, img_height - fovea_height
        # print(f'Valid limits: x: {valid_x_min} ~ {valid_x_max}, y: {valid_y_min} ~ {valid_y_max}')

        # 模拟退火迭代
        while temp > final_temp:
            counter += 1

            # 随机选择新的状态（邻域函数）
            if counter > 1:
                next_x_min = max(valid_x_min, current_x - next_pos_range * fovea_width)
                next_x_max = min(valid_x_max, current_x + next_pos_range * fovea_width)
                next_y_min = max(valid_y_min, current_y - next_pos_range * fovea_height)
                next_y_max = min(valid_y_max, current_y + next_pos_range * fovea_height)
                next_x = random.uniform(next_x_min, next_x_max)
                next_y = random.uniform(next_y_min, next_y_max)
            else:
                next_x, next_y = current_x, current_y
                # visualize_ann_result(tlwhs, next_x, next_y, fovea_width, fovea_height,
                #                       img_width, img_height,
                #                       save_path=visualize_path, img_name=f'ann_{marker}_initial.jpg')


            next_score = self.total_coverage(next_x, next_y, tlwhs, fovea_width, fovea_height,
                                        opt_version=opt_version, box_weight=box_weight)

            # 计算接受概率
            accept_probability = math.exp(min(5, (next_score - current_score) / temp))
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

    def optimize(self, tlwhs, fovea_width, fovea_height, img_width, img_height,
                init_x=None, init_y=None, epochs=1, algo='annealing',
                opt_version=1, box_weight=[],
                visualize=False, visualize_path='../results', visualize_mark='mark',
                dry_run=False):

        if dry_run:
            return img_width // 4, img_height // 4

        if algo == 'annealing':
            avg_x, avg_y, total_score = 0, 0, 0.0
            results = []
            for i in range(epochs):
                x, y, current_score = self.simulated_annealing(tlwhs, fovea_width, fovea_height, img_width, img_height,
                                        init_x=init_x, init_y=init_y,
                                        opt_version=opt_version, box_weight=box_weight,
                                        visualize=visualize, visualize_path='./debug/', marker=f'{visualize_mark}_epoch_{i + 1}')
                # print(f'Epoch {i+1} - Simulated annealing result x={x}, y={y}, score={current_score:.2f}')
                results.append((x, y, current_score))
                if visualize:
                    self.visualize_ann_result(tlwhs, x, y, fovea_width, fovea_height, img_width, img_height,
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
            # if visualize:
            #     self.visualize_ann_result(tlwhs, avg_x, avg_y, fovea_width, fovea_height,
            #                         img_width, img_height, save_path=visualize_path, img_name=f'test_avg.jpg',)
            
            return avg_x, avg_y
        else:
            return -1, -1


