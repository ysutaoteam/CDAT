#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import cv2
import xlrd
import scipy.io as scio
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore')


#读取健康人文件夹下的形式背景set

for path in os.listdir('E:/data/xingshibeijing/HC/1-FRFT-0'):
    data = pd.DataFrame(pd.read_excel(('E:/data/xingshibeijing/HC/1-FRFT-0'+"./"+path),header = 0,index_col=0))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
    # print(data)
    # jiaankangpinjie_matrix=[]   #用于邻接矩阵拼接形状为(192, 9, 9)的语句
    jiaankangpinjie_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，1728)的语句
    for jj in range(1, 253):
        huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
        result = huachunag_data.loc[huachunag_data['0°-20°'] == 1]  # 获取列名为0°-20°，内容为1的内容
        result2 = huachunag_data.loc[huachunag_data['20°-40°'] == 1]
        result3 = huachunag_data.loc[huachunag_data['40°-60°'] == 1]
        result4 = huachunag_data.loc[huachunag_data['60°-80°'] == 1]
        result5 = huachunag_data.loc[huachunag_data['80°-100°'] == 1]
        result6 = huachunag_data.loc[huachunag_data['100°-120°'] == 1]
        result7 = huachunag_data.loc[huachunag_data['120°-140°'] == 1]
        result8 = huachunag_data.loc[huachunag_data['140°-160°'] == 1]
        result9 = huachunag_data.loc[huachunag_data['160°-180°'] == 1]
        list = [result, result2, result3, result4, result5, result6, result7, result8, result9]
        adjacency_matrix = np.zeros((int(9), int(9)))
        for i in range(9):
            for j in range(9):
                if i == j:
                    adjacency_matrix[i][j] = len(list[i])  # min(len(list[i]), len(list[j]))
                elif set(list[i].index).intersection(set(list[j].index)) == set(list[i].index):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))
        
        #if   else用于将邻接矩阵拼接成(9，1728)的语句
        #作者：孙浩   2019.11.21
        
        if jj == 1:
            jiaankangpinjie_matrix = adjacency_matrix
        else:
            # # 垂直拼接
            # jiaankangpinjie_matrix = np.vstack((jiaankangpinjie_matrix, adjacency_matrix))
            # 横向拼接
            jiaankangpinjie_matrix = np.hstack((jiaankangpinjie_matrix, adjacency_matrix))
            # 或者（axis = 1 表示行,axis = 0 表示行列）
            # jiaankangpinjie_matrix= np.concatenate((jiaankangpinjie_matrix, adjacency_matrix),axis=1)  #水平组合

        # jiaankangpinjie_matrix.append(adjacency_matrix)   #.shape=(192, 9, 9)
    print(jiaankangpinjie_matrix)
    print(np.asarray(jiaankangpinjie_matrix).shape)
    filename = 'E:/data/mat-xingshibeijing/1_class/0'+"./"+path+'.mat'  # 保存的文件名
    scio.savemat(filename, {'1': jiaankangpinjie_matrix})  # 注意要以字典格式保存



#读取患病人文件夹下的形式背景set

for path in os.listdir('E:/data/xingshibeijing/PD/1-FRFT-0'):
    data = pd.DataFrame(pd.read_excel(('E:/data/xingshibeijing/PD/1-FRFT-0'+"./"+path),header = 0,index_col=0))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
    # print(data)
    # youbingpinjie_matrix=[]   #用于邻接矩阵拼接形状为(192, 9, 9)的语句
    youbingpinjie_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，1728)的语句
    for jj in range(1, 253):
        huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
        result = huachunag_data.loc[huachunag_data['0°-20°'] == 1]  # 获取列名为0°-20°，内容为1的内容
        result2 = huachunag_data.loc[huachunag_data['20°-40°'] == 1]
        result3 = huachunag_data.loc[huachunag_data['40°-60°'] == 1]
        result4 = huachunag_data.loc[huachunag_data['60°-80°'] == 1]
        result5 = huachunag_data.loc[huachunag_data['80°-100°'] == 1]
        result6 = huachunag_data.loc[huachunag_data['100°-120°'] == 1]
        result7 = huachunag_data.loc[huachunag_data['120°-140°'] == 1]
        result8 = huachunag_data.loc[huachunag_data['140°-160°'] == 1]
        result9 = huachunag_data.loc[huachunag_data['160°-180°'] == 1]
        list = [result, result2, result3, result4, result5, result6, result7, result8, result9]
        adjacency_matrix = np.zeros((int(9), int(9)))
        for i in range(9):
            for j in range(9):
                if i == j:
                    adjacency_matrix[i][j] = len(list[i])  # min(len(list[i]), len(list[j]))
                elif set(list[i].index).intersection(set(list[j].index)) == set(list[i].index):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))
        
       # if   else用于将邻接矩阵拼接成(9，1728)的语句
        #作者：孙浩   2019.11.21
       
        if jj == 1:
            youbingpinjie_matrix = adjacency_matrix
        else:
            # # 垂直拼接
            # youbingpinjie_matrix = np.vstack((youbingpinjie_matrix, adjacency_matrix))
            # 横向拼接
            youbingpinjie_matrix = np.hstack((youbingpinjie_matrix, adjacency_matrix))
            # 或者（axis = 1 表示行,axis = 0 表示行列）
            # youbingpinjie_matrix= np.concatenate((youbingpinjie_matrix, adjacency_matrix),axis=1)  #水平组合

        # youbingpinjie_matrix.append(adjacency_matrix)   #.shape=(192, 9, 9)
    print(youbingpinjie_matrix)
    print(np.asarray(youbingpinjie_matrix).shape)
    filename = 'E:/data/mat-xingshibeijing/0_class/0' + "./" + path + '.mat'  # 保存的文件名
    scio.savemat(filename, {'0': youbingpinjie_matrix})  # 注意要以字典格式保存



#做CPPDD数据集时下面可先注释掉    下面的代码可能是无用的、禁止使用
'''
读取测试文件夹下的形式背景set
'''
# for path in os.listdir('F:/HOG-Picture/set-xingshibeijingshengcheng/test'):
#     data = pd.DataFrame(pd.read_excel(('F:/HOG-Picture/set-xingshibeijingshengcheng/test'+"./"+path),header = 0,index_col=0))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
#     # print(data)
#     # testpinjie_matrix=[]   #用于邻接矩阵拼接形状为(192, 9, 9)的语句
#     testpinjie_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，1728)的语句
#     for jj in range(1, 193):
#         huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
#         result = huachunag_data.loc[huachunag_data['0°-20°'] == 1]  # 获取列名为0°-20°，内容为1的内容
#         result2 = huachunag_data.loc[huachunag_data['20°-40°'] == 1]
#         result3 = huachunag_data.loc[huachunag_data['40°-60°'] == 1]
#         result4 = huachunag_data.loc[huachunag_data['60°-80°'] == 1]
#         result5 = huachunag_data.loc[huachunag_data['80°-100°'] == 1]
#         result6 = huachunag_data.loc[huachunag_data['100°-120°'] == 1]
#         result7 = huachunag_data.loc[huachunag_data['120°-140°'] == 1]
#         result8 = huachunag_data.loc[huachunag_data['140°-160°'] == 1]
#         result9 = huachunag_data.loc[huachunag_data['160°-180°'] == 1]
#         list = [result, result2, result3, result4, result5, result6, result7, result8, result9]
#         adjacency_matrix = np.zeros((int(9), int(9)))
#         for i in range(9):
#             for j in range(9):
#                 if i == j:
#                     adjacency_matrix[i][j] = len(list[i])  # min(len(list[i]), len(list[j]))
#                 elif set(list[i].index).intersection(set(list[j].index)) == set(list[i].index):
#                     adjacency_matrix[i][j] = 0
#                 else:
#                     adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))
#         '''
#         if   else用于将邻接矩阵拼接成(9，1728)的语句
#         作者：孙浩   2019.11.21
#         '''
#         if jj == 1:
#             testpinjie_matrix = adjacency_matrix
#         else:
#             # # 垂直拼接
#             # testpinjie_matrix = np.vstack((testpinjie_matrix, adjacency_matrix))
#             # 横向拼接
#             testpinjie_matrix = np.hstack((testpinjie_matrix, adjacency_matrix))
#             # 或者（axis = 1 表示行,axis = 0 表示行列）
#             # testpinjie_matrix= np.concatenate((testpinjie_matrix, adjacency_matrix),axis=1)  #水平组合
#
#         # testpinjie_matrix.append(adjacency_matrix)   #.shape=(192, 9, 9)
#     print(testpinjie_matrix)
#     print(np.asarray(testpinjie_matrix).shape)
#     filename = 'F:/HOG-Picture/train/test' + "./" + path + '.mat'  # 保存的文件名
#     scio.savemat(filename, {'2': testpinjie_matrix})  # 注意要以字典格式保存
#
