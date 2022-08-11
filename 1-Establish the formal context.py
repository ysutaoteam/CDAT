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
        Constructing functions (构造函数) 
            Default parameter, a block consists of 2x2 cells, and the step size is 1 cell.(默认参数，一个block由2x2个cell组成，步长为1个cell大小)
        args:
            img：input grayscale image (输入灰度图像)
            cell_size：the size of a cell unit (细胞单元的大小)
            bin_size：the number of bins in the histogram (直方图的bin个数)
        '''
        self.img = img
        '''
        The color space of the input image is normalized using the Gamma correction method to adjust the contrast of the image.(采用Gamma校正法对输入图像进行颜色空间的标准化（归一化），目的是调节图像的对比度)
        '''
        self.img = np.sqrt(img * 1.0 / float(np.max(img)))
        self.img = self.img * 255
        # print('img',self.img.dtype)   #float64
        # initialize the parameters (初始化参数) 
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 180 / self.bin_size     #  The angle range is set to 180. (角度范围设为180度)
        assert type(self.bin_size) == int, "bin_size should be integer,"
        assert type(self.cell_size) == int, "cell_size should be integer,"
        assert 180 % self.bin_size == 0, "bin_size should be divisible by 180"   

    def extract(self):
        
        height, width = self.img.shape

        '''
        1、Calculate the magnitude and angle of gradient of each pixel. (计算每一个像素点的梯度幅值和角度)
        '''
        gradient_magnitude, gradient_angle = self.global_gradient()  # Return the magnitude and angle of gradient of each pixel  返回每一像素点的梯度大小和角度值
        #  Calculate the absolute value of the gradient. (梯度取绝对值)
        gradient_magnitude = abs(gradient_magnitude)

        '''
        2、Calculate the gradient histogram for each cell of the input image. (计算输入图像的每个cell单元的梯度直方图)
        '''
        m = 0
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))   
        # iterate over rows and columns (遍历行和列)
        for i in range(cell_gradient_vector.shape[0]):     
            for j in range(cell_gradient_vector.shape[1]):   
                # Calculate the feature vector of the [i][j]-th cell (计算第[i][j]个cell的特征向量)
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,      
                                 j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                             j * self.cell_size:(j + 1) * self.cell_size]
                m = m + 1   # The parameter m is used to display the order of cells in cell_gradient (参数m用于显示cell_gradient中的第m个细胞)
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle,m)      
        print(m)

        # Draw the gradient direction histogram of each cell obtained to obtain the feature map. (将得到的每个cell的梯度方向直方图绘出，得到特征图)
        hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)    

        '''
        3、Concatenate 2x2 cells into a block, and concatenate the features of all cells in a block to get thefeature descriptor of the block.
           The feature descriptors of all blocks are concatenated to get the HOG feature descriptor of the image, which is the features for the final classification.
           (将2x2个cell组成一个block，一个block内所有cell的特征串联起来得到该block的特征描述子,将所有block的特征描述子串联起来得到该图像的特征描述子，
           这就是最终用于分类的特征向量)
        '''
        jishu = 0     # jishu is the number of block. (jishu是block的数量)
        hog_vector = []   # hog_vector is used to store the statistics of each block (hog_vector用来存储每个block的统计信息)
        # The default step size is a cell size. A block consists of 2x2 cells, and each block is traversed.(默认步长为一个cell大小，一个block由2x2个cell组成，遍历每一个block)
        for i in range(cell_gradient_vector.shape[0] - 1):
            for j in range(cell_gradient_vector.shape[1] - 1):
                # Extract the feature of the [i][j]-th block. (提取第[i][j]个block的特征向量)
                block_vector = []
                block_vector.extend(cell_gradient_vector[i][j])
                block_vector.extend(cell_gradient_vector[i][j + 1])
                block_vector.extend(cell_gradient_vector[i + 1][j])
                block_vector.extend(cell_gradient_vector[i + 1][j + 1])
              
                # Calculate the l2 norm.(计算l2范数)
                mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
                magnitude = mag(block_vector) + 1e-5
                # Normalization. (归一化)
                if magnitude != 0:
                    normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                    block_vector = normalize(block_vector, magnitude)
                hog_vector.append(block_vector)
                jishu =jishu+1
        return np.asarray(hog_vector), hog_image

    def global_gradient(self):
        '''
        Calculate the gradient of the image along the x-axis and y-axis separately.(分别计算图像沿x轴和y轴的梯度)
        '''
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)      #dx
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)      #dy
       
        # Calculate the magnitude of gradient.(计算梯度幅值)
        # gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
       
        # Calculate the angle of gradient.(计算梯度方向)
        # gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        '''
        Returns the magnitude and angle of gradient of each pixel.(返回每一个像素点的梯度大小和角度值)
        '''
        gradient_magnitude, gradient_angle = cv2.cartToPolar(gradient_values_x, gradient_values_y, angleInDegrees=True)

        # If the angle is greater than 180°, then subtract 180°.(角度大于180°的，减去180度)
        gradient_angle[gradient_angle > 180.0] -= 180
        # print('gradient',gradient_magnitude.shape,gradient_angle.shape,np.min(gradient_angle),np.max(gradient_angle))
        return gradient_magnitude, gradient_angle


    def cell_gradient(self, cell_magnitude, cell_angle,m):
        '''
        Build a histogram of gradient directions for each cell unit.(为每个细胞单元构建梯度方向直方图)

        args:
            cell_magnitude：The magnitude of the gradient of each pixel in the cell. (cell中每个像素点的梯度幅值)
            cell_angle：The angle of the gradient of each pixel in the cell. (cell中每个像素点的梯度方向)
        return：
            Returns the gradient histogram corresponding to the cell, and the length is bin_size.(返回该cell对应的梯度直方图，长度为bin_size)
        '''
        d2 = {}                # d2 is Used to store all dictionaries in a cell. (d2用于存储细胞中所有字典)
        kk = 0                # kk is used to traverse all the pixels in a cell. (用来计算一个细胞64个像素的遍历)
        orientation_centers = [0] * self.bin_size            #Build an array [0, 0, 0, 0, 0, 0, 0, 0, 0] for storage. (构建[0, 0, 0, 0, 0, 0, 0, 0, 0]进行存储)
              
        # Traverse all the pixels in a cell. (遍历cell中的每一个像素点)
        for i in range(cell_magnitude.shape[0]):         #cell_magnitude=（8，8）
            for j in range(cell_magnitude.shape[1]):
                # Magnitude of the gradient. (梯度幅值)
                gradient_strength = cell_magnitude[i][j]
                # Angle of the gradient      (梯度方向)
                gradient_angle = cell_angle[i][j]
                # Bilinear interpolation (双线性插值)
                min_angle, max_angle, weight = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - weight))
                orientation_centers[max_angle] += (gradient_strength * weight)

                kk = kk + 1
                if kk < 65:
                    # Create a dictionary d1 to store the min_angle and max_angle of the pixels. (创建字典d1来存储所有像素点的min_angle 和 max_angle)
                    d1 = {}
                    key = kk
                    value = min_angle
                    d1.setdefault(key, []).append(value)
                    value2 = max_angle
                    d1.setdefault(key, []).append(value2)

                else:
                    kk = 0
                d2.update(d1)  
                '''
                Write data to the table to establish the formal context. (将数据写入到表格中，形成形式背景)
                                
                '''
                worksheet.write(kk+(64*(m-1)),0,kk)
                worksheet.write(kk + (64 * (m - 1)),min_angle+1, 1)
                worksheet.write(kk + (64 * (m - 1)),max_angle+1,1)
       
        workbook.save('E:/data/xingshibeijing/PD/1-FRFT-3' + "./" + filename + '.xls')

        print('the statistical histogram of the'+str(m)+'cell：',orientation_centers)   # Output the statistical histogram of the str(m)-th cell. (输出第str(m)个细胞的统计直方图)
        return orientation_centers


    def get_closest_bins(self, gradient_angle):
        '''
        Bilinear interpolation is used to calculate which bin the gradient_angle is located in.  (计算梯度方向gradient_angle位于哪一个bin中，这里采用的计算方式为双线性插值)

        args:
            gradient_angle:angle
        return：
            start,end,weight：Start index of bin, end index of bin, the weight of the bin corresponding to the end index. (起始bin索引，终止bin的索引，end索引对应bin所占权重)
        '''
        idx = int(gradient_angle / self.angle_unit)   #angle_unit=20
        mod = gradient_angle % self.angle_unit
        return idx % self.bin_size, (idx + 1) % self.bin_size, mod / self.angle_unit

    def render_gradient(self, image, cell_gradient):
        '''
        Plot the gradient direction histogram of each cell to obtain the feature map. (将得到的每个cell的梯度方向直方图绘出，得到特征图)
        args：
            image：a canvas, as big as the input image (画布,和输入图像一样大)
            cell_gradient：Gradient histogram [h/cell_size,w/cell_size,bin_size] for each cell of the input image (输入图像的每个cell单元的梯度直方图,形状为[h/cell_size,w/cell_size,bin_size])
        return：
            image：feature map (特征图)
        '''
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        # iterate over each cell (遍历每一个cell)
        for x in range(cell_gradient.shape[0]):
            for y in range(cell_gradient.shape[1]):
                # Obtain the gradient histogram of the [i][j]-th cell (获取第[i][j]个cell的梯度直方图)
                cell_grad = cell_gradient[x][y]
                # Normalization (归一化)
                cell_grad /= max_mag
                angle = 0
                angle_gap = self.angle_unit
                # Traverse each bin interval (遍历每一个bin区间)
                for magnitude in cell_grad:
                    # Convert to radian (转换为弧度)
                    angle_radian = math.radians(angle)
                    # Calculate the start and end coordinates, and the length is the magnitude(normalized). (计算起始坐标和终点坐标，长度为幅值(归一化))
                    x1 = int(x * self.cell_size + cell_width + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + cell_width + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size + cell_width - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size + cell_width - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_gap
        return image


if __name__ == '__main__':
    # load image (加载图像)
    for filename in os.listdir('E:/data/yuputu_/PD/1-FRFT-3'):  #'F:/HOG-Picture/HealthyPicture'      PatientPicture   TestPicture healthyVoice-jiequ  pantientvice-jiequ
        img = cv2.imread('E:/data/yuputu_/PD/1-FRFT-3' + "./" + filename)

        # img = cv2.imread('./yuputu.png')
        # print('read image',img)
        width = 64
        height = 192
        img_copy = img[:, :, ::-1]
        gray_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # Create workbook (创建workbook)
        workbook = xlwt.Workbook(encoding='utf-8')
        # Create table (创建表)
        worksheet = workbook.add_sheet('sheet1')
        # Write content to cell of table: write header (往表中的单元格内写入内容:写入表头)
        worksheet.write(0, 0, label="pixels")
        worksheet.write(0, 1, label="0°-20°")
        worksheet.write(0, 2, label="20°-40°")
        worksheet.write(0, 3, label="40°-60°")
        worksheet.write(0, 4, label="60°-80°")
        worksheet.write(0, 5, label="80°-100°")
        worksheet.write(0, 6, label="100°-120°")
        worksheet.write(0, 7, label="120°-140°")
        worksheet.write(0, 8, label="140°-160°")
        worksheet.write(0, 9, label="160°-180°")

        # Original image (显示原图像)
        plt.figure(figsize=(6.4, 2.0 * 3.2))
        plt.subplot(1, 2, 1)
        plt.imshow(img_copy)

        # Feature extraction (特征提取)
        hog = Hog_descriptor(gray_copy, cell_size=8, bin_size=9)
        hog_vector, hog_image = hog.extract()
        print('hog_vector', hog_vector.shape)
        print(hog_vector)
        print('hog_image', hog_image.shape)
        print(hog_image)

'''
       # Plot feature map (绘制特征图)
        plt.subplot(1, 2, 2)
        plt.imshow(hog_image, cmap=plt.cm.gray)
        plt.show()

'''





