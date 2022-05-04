import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.feature import hog
import networkx as nx
from copy import deepcopy
import pandas as pd
import xlsxwriter
import xlwt
import openpyxl


class Hog_descriptor():
       def __init__(self, img, cell_size=8, bin_size=9):
        '''
        构造函数
            默认参数，一个block由2x2个cell组成，步长为1个cell大小
        args:
            img：输入图像(更准确的说是检测窗口)，这里要求为灰度图像  对于行人检测图像大小一般为128x64 即是输入图像上的一小块裁切区域
            cell_size：细胞单元的大小 如8，表示8x8个像素
            bin_size：直方图的bin个数
        '''
        self.img = img
        '''
        采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），目的是调节图像的对比度，降低图像局部
        的阴影和光照变化所造成的影响，同时可以抑制噪音。采用的gamma值为0.5。 f(I)=I^γ
        '''
        self.img = np.sqrt(img * 1.0 / float(np.max(img)))
        self.img = self.img * 255
        # print('img',self.img.dtype)   #float64
        # 参数初始化
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 180 / self.bin_size  # 这里采用180°
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert 180 % self.bin_size == 0, "bin_size should be divisible by 180"    #  %是取余数的运算

    def extract(self):
        
        height, width = self.img.shape

        '''
        1、计算图像每一个像素点的梯度幅值和角度
        '''
        gradient_magnitude, gradient_angle = self.global_gradient()  #返回每一像素点的梯度大小和角度值
        #梯度取绝对值
        gradient_magnitude = abs(gradient_magnitude)

        '''
        2、计算输入图像的每个cell单元的梯度直方图，形成每个cell的descriptor 比如输入图像为192x64 可以得到24x8=192个cell，每个cell由9个bin组成
        '''
        m = 0
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))   #形成一个(94,76.9)大小数组
        # 遍历每一行、每一列
        for i in range(cell_gradient_vector.shape[0]):       #cell_gradient_vector.shape[0] =94
            for j in range(cell_gradient_vector.shape[1]):   #cell_gradient_vector.shape[1] =76   cell_gradient_vector=7144
                # 计算第[i][j]个cell的特征向量    每一个[i][j]中存8*8个像素点的梯度值和角度值
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,      #   切片  例如【0*8：1*8】输出0-8之间的值
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                m = m + 1   # 在cell_gradient中在传一个参数m,用于cell_gradient中显示第几个细胞
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle,m)       #cell_gradient_vector.shape=(94,76.9)
        print(m)

        # 将得到的每个cell的梯度方向直方图绘出，得到特征图（特征图就是hog可视化图）
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)     #hog_image.shape=(752，608)

        '''
        3、将2x2个cell组成一个block，一个block内所有cell的特征串联起来得到该block的HOG特征descriptor
           将图像image内所有block的HOG特征descriptor串联起来得到该image（检测目标）的HOG特征descriptor，
           这就是最终分类的特征向量
        '''
        jishu = 0     #jishu=6975=93*75  为块的数目
        hog_vector = []   #用来存储每个块的统计信息，两两之间串联关系
        # 默认步长为一个cell大小，一个block由2x2个cell组成，遍历每一个block
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                # 提取第[i][j]个block的特征向量
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
                '''块内归一化梯度直方图，去除光照、阴影等变化，增加鲁棒性'''
                # 计算l2范数
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector) + 1e-5
                # 归一化
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
                jishu =jishu+1
        return np.asarray(hog_vector), hog_image

    def global_gradient(self):
        '''
        分别计算图像沿x轴和y轴的梯度
        '''
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)      #dx
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)      #dy
        # 计算梯度幅值 这个计算的是0.5*gradient_values_x + 0.5*gradient_values_y
        # gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        # 计算梯度方向
        # gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        '''
        返回每一个像素点的梯度大小和角度值
        '''
        gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)

        # 角度大于180°的，减去180度
        gradient_angle[gradient_angle > 180.0] -= 180
        # print('gradient',gradient_magnitude.shape,gradient_angle.shape,np.min(gradient_angle),np.max(gradient_angle))
        return gradient_magnitude, gradient_angle


    def cell_gradient(self, cell_magnitude, cell_angle,m):
        '''
        为每个细胞单元构建梯度方向直方图

        args:
            cell_magnitude：cell中每个像素点的梯度幅值
            cell_angle：cell中每个像素点的梯度方向
        return：
            返回该cell对应的梯度直方图，长度为bin_size
        '''
        d2 = {}  #用于存储细胞中所有字典  64个
        kk = 0 #用来计算一个细胞64个像素的遍历
        orientation_centers = [0] * self.bin_size   #构建[0, 0, 0, 0, 0, 0, 0, 0, 0]进行存储
        # 遍历cell中的每一个像素点
        for i in range(cell_magnitude.shape[0]):         #cell_magnitude=（8，8）
            for j in range(cell_magnitude.shape[1]):
                # 梯度幅值
                gradient_strength = cell_magnitude[i][j]
                # 梯度方向
                gradient_angle = cell_angle[i][j]
                # 双线性插值
                min_angle, max_angle, weight = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - weight))
                orientation_centers[max_angle] += (gradient_strength * weight)
                # orientation_centers[min_angle] += 1
                # orientation_centers[max_angle] += 1
                kk = kk + 1
                if kk < 65:
                    #字典的一键多值
                    # 创建一个字典，用于存储第几个像素  计算的min_anfle  max_amgle   dict={key1:[value1,value2]}
                    d1 = {}
                    key = kk
                    value = min_angle
                    d1.setdefault(key, []).append(value)
                    value2 = max_angle
                    d1.setdefault(key, []).append(value2)
                    # dic = deepcopy(d1)
                    # # d = {}
                    # d = dict(d1,**dic)
                else:
                    kk = 0
                d2.update(d1)   #每一个像素点为为一个字典d1，将d1给d2，每次在d2后面更新d1
                '''
                将数据写入到表格中，形成形式背景，这样的话，上面的字典可能不需要了，只需要kk   min_angle  max_angle。
                
                '''
                worksheet.write(kk+(64*(m-1)),0,kk)
                worksheet.write(kk + (64 * (m - 1)),min_angle+1, 1)
                worksheet.write(kk + (64 * (m - 1)),max_angle+1,1)
       
        workbook.save('E:/data/xingshibeijing/PD/1-FRFT-3' + "./" + filename + '.xls')

        print('第'+str(m)+'个细胞的统计直方图：',orientation_centers)
        return orientation_centers


    def get_closest_bins(self, gradient_angle):
        '''
        计算梯度方向gradient_angle位于哪一个bin中，这里采用的计算方式为双线性插值

        args:
            gradient_angle:角度
        return：
            start,end,weight：起始bin索引，终止bin的索引，end索引对应bin所占权重
        '''
        idx = int(gradient_angle / self.angle_unit)   #angle_unit=20
        mod = gradient_angle % self.angle_unit
        return idx % self.bin_size, (idx + 1) % self.bin_size, mod / self.angle_unit

    def render_gradient(self, image, cell_gradient):
        '''
        将得到的每个cell的梯度方向直方图绘出，得到特征图
        args：
            image：画布,和输入图像一样大 [h,w]
            cell_gradient：输入图像的每个cell单元的梯度直方图,形状为[h/cell_size,w/cell_size,bin_size]
        return：
            image：特征图（hog可视化图）
        '''
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        # 遍历每一个cell
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                # 获取第[i][j]个cell的梯度直方图
                cell_grad = cell_gradient[x][y]
                # 归一化
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                # 遍历每一个bin区间
                for magnitude in cell_grad:
                    # 转换为弧度
                    angle_radian = math.radians(angle)
                    # 计算起始坐标和终点坐标，长度为幅值(归一化),幅值越大、绘制的线条越长、越亮
                    x1 = int(x * self.cell_size + cell_width + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + cell_width + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + cell_width - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + cell_width - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


if __name__ == '__main__':
    # 加载图像
    for filename in os.listdir('E:/data/yuputu_/PD/1-FRFT-3'):  #'F:/HOG-Picture/HealthyPicture'      PatientPicture   TestPicture healthyVoice-jiequ  pantientvice-jiequ
        img = cv2.imread('E:/data/yuputu_/PD/1-FRFT-3' + "./" + filename)

        # img = cv2.imread('./yuputu.png')
        # print('读取到的图像',img)
        width = 64
        height = 192
        img_copy = img[:, :, ::-1]
        gray_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # 创建workbook（其实就是excel，后来保存一下就行）
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建表
        worksheet = workbook.add_sheet('sheet1')
        # 往单元格内写入内容:写入表头
        worksheet.write(0, 0, label="像素点")
        worksheet.write(0, 1, label="0°-20°")
        worksheet.write(0, 2, label="20°-40°")
        worksheet.write(0, 3, label="40°-60°")
        worksheet.write(0, 4, label="60°-80°")
        worksheet.write(0, 5, label="80°-100°")
        worksheet.write(0, 6, label="100°-120°")
        worksheet.write(0, 7, label="120°-140°")
        worksheet.write(0, 8, label="140°-160°")
        worksheet.write(0, 9, label="160°-180°")

        # 显示原图像
        plt.figure(figsize=(6.4, 2.0 * 3.2))
        plt.subplot(1, 2, 1)
        plt.imshow(img_copy)

        # HOG特征提取
        hog = Hog_descriptor(gray_copy, cell_size=8, bin_size=9)
        hog_vector, hog_image = hog.extract()
        print('hog_vector', hog_vector.shape)
        print(hog_vector)
        print('hog_image', hog_image.shape)
        print(hog_image)

'''
       # 绘制特征图
        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap=plt.cm.gray)
        plt.show()

'''





