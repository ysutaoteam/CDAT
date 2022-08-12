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


# read formal context under healthy people folder (读取健康人文件夹下的形式背景)

for path in os.listdir('E:/data/xingshibeijing/HC/1-FRFT-0'):
    data = pd.DataFrame(pd.read_excel(('E:/data/xingshibeijing/HC/1-FRFT-0'+"./"+path),header = 0,index_col=0))  # To read data, set None to generate a dictionary. The key value in the dictionary is the sheet name. (读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字。)

    jiaankangpinjie_matrix = np.zeros((int(9), int(9))) 
    for jj in range(1, 253):
        huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
        result = huachunag_data.loc[huachunag_data['0°-20°'] == 1]  # Get the content whose column name is 0°-20° and the content is 1. (获取列名为0°-20°，内容为1的内容。)
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
        
        
        if jj == 1:
            jiaankangpinjie_matrix = adjacency_matrix
        else:
            # vertical stitching (垂直拼接)
            # jiaankangpinjie_matrix = np.vstack((jiaankangpinjie_matrix, adjacency_matrix))
            # Horizontal stitching (横向拼接)
            jiaankangpinjie_matrix = np.hstack((jiaankangpinjie_matrix, adjacency_matrix))
            # axis = 1 represents row, axis = 0 represents column(axis = 1 表示行,axis = 0 表示列)
            # jiaankangpinjie_matrix= np.concatenate((jiaankangpinjie_matrix, adjacency_matrix),axis=1)  # horizontal combination (水平组合)

        # jiaankangpinjie_matrix.append(adjacency_matrix)   #.shape=(192, 9, 9)
    print(jiaankangpinjie_matrix)
    print(np.asarray(jiaankangpinjie_matrix).shape)
    filename = 'E:/data/mat-xingshibeijing/1_class/0'+"./"+path+'.mat'  # saved filename (保存的文件名)
    scio.savemat(filename, {'1': jiaankangpinjie_matrix})  # save in dictionary format (注意要以字典格式保存)



# read formal context under healthy people folder (读取患病人文件夹下的形式背景)

for path in os.listdir('E:/data/xingshibeijing/PD/1-FRFT-0'):
    data = pd.DataFrame(pd.read_excel(('E:/data/xingshibeijing/PD/1-FRFT-0'+"./"+path),header = 0,index_col=0))
    # print(data)
    # youbingpinjie_matrix=[]  
    youbingpinjie_matrix = np.zeros((int(9), int(9)))  
    for jj in range(1, 253):
        huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
        result = huachunag_data.loc[huachunag_data['0°-20°'] == 1]  
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

       
        if jj == 1:
            youbingpinjie_matrix = adjacency_matrix
        else:

            # youbingpinjie_matrix = np.vstack((youbingpinjie_matrix, adjacency_matrix))

            youbingpinjie_matrix = np.hstack((youbingpinjie_matrix, adjacency_matrix))

            # youbingpinjie_matrix= np.concatenate((youbingpinjie_matrix, adjacency_matrix),axis=1) 

        # youbingpinjie_matrix.append(adjacency_matrix)   #.shape=(192, 9, 9)
    print(youbingpinjie_matrix)
    print(np.asarray(youbingpinjie_matrix).shape)
    filename = 'E:/data/mat-xingshibeijing/0_class/0' + "./" + path + '.mat'  
    scio.savemat(filename, {'0': youbingpinjie_matrix}) 

