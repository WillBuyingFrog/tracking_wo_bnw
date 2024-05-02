import cv2
import argparse
import numpy as np
import os



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


class FrogROI:
    def __init__(self, image_width, image_height, init_image_path,
                 region_scale=0.1, pixel_change_threshold=50, is_PIL=False):
        self.image_width = image_width
        self.image_height = image_height
        self.init_image_path = init_image_path
        self.is_PIL = is_PIL
        # self.current_image = cv2.imread(init_image_path)
        # 需要持续更新的比较基准（engram）是和current_image形状相同的numpy array，初始值全部置零
        if is_PIL:
            self.engram = np.zeros((3, self.image_height, self.image_width), dtype=np.float32)
        else:
            self.engram = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)
        # self.engram_factor = np.array([5, 3, 2.5, 2, 1.5, 1.25, 1, 0.75, 0.5, 0.25])
        # self.engram_factor = np.array([10, 2.5, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1])
        # Inspired from "Predictive properties of visual adaptation"
        self.engram_factor = np.array([10, 5, 2.5, 1, 0.5, 0.2, 0.2, -0.2, -0.5, -1])
        # The engram factors below doesnt work :(
        # self.engram_factor = np.array([2.5, 2.5, 5.0, 7.5, 7.5, 7.5, 5.0, 1.5, 0.5, 0.2])
        self.region_scale = region_scale
        self.pixel_change_threshold = pixel_change_threshold
        # 用于计算比较基准的图片
        self.engram_images = []

    
    def store_engram_images(self, image_path):
        self.engram_images.append(cv2.imread(image_path))
        if len(self.engram_images) > 10:
            self.engram_images.pop(0)

    def store_engram_images_raw(self, raw_image):
        self.engram_images.append(raw_image)
        if len(self.engram_images) > 10:
            self.engram_images.pop(0)
    
    def calculate_engram(self):
        num_images = len(self.engram_images)
        factor_sum = 0.0
        for i in range(num_images):
            self.engram += self.engram_images[i] * self.engram_factor[i]
            factor_sum += self.engram_factor[i]

        self.engram = self.engram / factor_sum
    
    def compare_engram(self, target_image,
                       visualize=False, visualize_path="results/", marker=''):
        
        if visualize:
            logger = Logger(os.path.join(visualize_path, f"engram_{marker}_log.txt"))

        # 契合行人目标的长宽比的区域划分？
        region_w = int(self.image_width * self.region_scale)
        region_h = int(self.image_height * self.region_scale * 3)
        num_regions_w = self.image_width // region_w
        num_regions_h = self.image_height // region_h

        threshold = region_h * region_w * self.pixel_change_threshold

        if self.is_PIL:
            threshold /= 255

        if visualize:
            logger.write_log(f'threshold: {threshold}')

        detected_regions = []
        detected_regions_diff = []

        # 目标图片和engram的差异
        diff = cv2.absdiff(self.engram, target_image)

        for i in range(num_regions_w):
            for j in range(num_regions_h):
                if self.is_PIL:
                    region_diff = diff[:, j*region_h:(j+1)*region_h, i*region_w:(i+1)*region_w]
                else:
                    region_diff = diff[j*region_h:(j+1)*region_h, i*region_w:(i+1)*region_w]
                region_diff_sum = region_diff.sum()
                if region_diff_sum > threshold:
                # print(f'Region ({i}, {j}) diff sum: {region_diff_sum}')
                    # 计算这个有突变的区域的tlwh，然后加入detected_regions
                    # t: 该区域的起始x坐标。l: 该区域的起始y坐标。w: 该区域的宽度。h: 该区域的高度。
                    detected_regions.append((i*region_w, j*region_h, region_w, region_h))
                    detected_regions_diff.append(region_diff_sum)
        
        # 如果要可视化，就在图上画出所有探测到的区域
        if visualize:
            for index, region in enumerate(detected_regions):
                # if marker == '2':
                #     # 对第二轮比较，把每个检测到的区域对应的原图部分单独拿出来并保存为小图片
                #     cv2.imwrite(os.path.join(visualize_path, f"image_{marker}_target_{region[0]}_{region[1]}.jpg"), target_image[region[1]:region[1]+region[3], region[0]:region[0]+region[2]])
                #     # 把痕迹的对应部分也拿出来保存为小图片
                #     cv2.imwrite(os.path.join(visualize_path, f"image_{marker}_engram_{region[0]}_{region[1]}.jpg"), self.engram[region[1]:region[1]+region[3], region[0]:region[0]+region[2]])
                #     # 加载这两张刚保存的小图片，计算它们的差异
                #     target_region = cv2.imread(os.path.join(visualize_path, f"image_{marker}_target_{region[0]}_{region[1]}.jpg"))
                #     engram_region = cv2.imread(os.path.join(visualize_path, f"image_{marker}_engram_{region[0]}_{region[1]}.jpg"))
                #     diff_region = cv2.absdiff(engram_region, target_region)
                #     logger.write_log(f'Diff of region {region}: {diff_region.sum()}')
                #     logger.write_log(f'raw target region data: {target_region}')
                #     logger.write_log(f'raw engram region data: {engram_region}')

                cv2.rectangle(target_image, (region[0], region[1]), (region[0]+region[2], region[1]+region[3]), (0, 255, 0), 2)
                logger.write_log(f'Region {region}, diff {detected_regions_diff[index]}')
            cv2.imwrite(os.path.join(visualize_path, f"image_{marker}_engram.jpg"), self.engram)
            cv2.imwrite(os.path.join(visualize_path, f"image_{marker}_target.jpg"), target_image)
            logger.close()
        return detected_regions

    
def blur_image(img, blur_factor):
        return cv2.GaussianBlur(img, (blur_factor, blur_factor), 0)


def visualize_rois(img, img0, engram, roi_tlwhs, result_path='results/', file_marker='test'):
    for roi in roi_tlwhs:
        cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(result_path, f"{file_marker}_roi.jpg"), img)
    cv2.imwrite(os.path.join(result_path, f"{file_marker}_origin.jpg"), img0)
    cv2.imwrite(os.path.join(result_path, f"{file_marker}_engram.jpg"), engram)

def visualize_all_regions(img, roi_tlwhs, target_tlwhs, result_path='results/', file_marker='test'):
    for roi in roi_tlwhs:
        cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
    for target in target_tlwhs:
        print(f'Drawing target: {target}')
        top_left = (int(target[0]), int(target[1]))
        right_bottom = (int(target[0] + target[2]), int(target[1] + target[3]))
        cv2.rectangle(img, top_left, right_bottom, (255, 0, 0), 2)
    cv2.imwrite(os.path.join(result_path, f"{file_marker}_all_regions.jpg"), img)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', default='image', type=str, help='测试数据类型。如果是dir就代表像多目标跟踪模型里一样，按顺序计算所有帧图片的感兴趣区域')
    parser.add_argument('--data_dir', default='roi_images/', type=str, help='测试数据存放的位置')
    parser.add_argument('--region_scale', default=0.025, type=float, help='感兴趣区域的比例系数')
    parser.add_argument('--pixel_change_threshold', default=70, type=int, help='感兴趣区域的像素变化阈值')
    parser.add_argument('--debug', default=False, type=bool, help='是否开启debug输出')

    args = parser.parse_args()
    test_type = args.test_type
    data_dir = args.data_dir
    region_scale = args.region_scale
    pixel_change_threshold = args.pixel_change_threshold
    debug = args.debug

    if test_type == 'image':
        # 从data_dir中加载十五张图片（保证data_dir中有且只有十五张图片），
        # 按照文件名升序排序，依次存入images_path中
        images_path = []

        if debug:
            print(f"Loading images from {data_dir}...")

        for file in os.listdir(data_dir):
            # 如果文件是jpg格式的图片，就加入images_path
            if file.endswith('.jpg'):
                origin_image_path = os.path.join(data_dir, file)
                # 并将这张图片做模糊处理，模糊后的图片保存到roi_blurred_images文件夹下
                img = cv2.imread(origin_image_path)
                img = blur_image(img, 37)
                # 为了计算方便，图像长宽各缩小到原来的一半
                img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
                
                # 在加载图片路径时便一同获取经过缩小后的图片宽高
                image_height, image_width = img.shape[:2]

                blurred_image_path = os.path.join(data_dir, 'blur', file)
                if debug:
                    print(f'Adding {blurred_image_path} to images_path...')
                images_path.append(blurred_image_path)
                cv2.imwrite(blurred_image_path, img)
        if debug:
            print(f'Size of blurred images: {image_height} x {image_width}')
                    

        init_image_path = images_path[0]
        
        
        region_detector = FrogROI(image_height=image_height, image_width=image_width, init_image_path=init_image_path,
                                  region_scale=region_scale, pixel_change_threshold=pixel_change_threshold)
        for i in range(1, len(images_path)):
            current_image = cv2.imread(images_path[i])
            region_detector.store_engram_images(images_path[i])
            region_detector.calculate_engram()
            detected_regions = region_detector.compare_engram(current_image.astype(np.float32), 
                                                              visualize=True, visualize_path="results/", marker=str(i))
            # print(detected_regions)