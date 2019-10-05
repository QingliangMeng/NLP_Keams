#-*- coding:utf-8 -*-
import os
import numpy as np
from sys import argv
path = argv[1]
dir_path = os.listdir(path)
res_list = []
for i in dir_path:
    file_path = i
    numb_list = []
    part_res = []
    numb2_list = []
    with open (path + '/' + file_path , 'r') as f:
        first_line_count = 0
        for line in f:
            if first_line_count == 0:
                cluster_distance = float(line.split(' = ')[1])
                first_line_count += 1
                continue
            numb = len(line)
            part_res.append(line)
            numb_list.append(numb)
    u = np.mean(numb_list)
    std = np.std(numb_list)
    second_line_list = list(set(part_res))
    for line2 in second_line_list:
        numb2 = len(line2)
        numb2_list.append(numb2)
    
    u2 = np.mean(numb2_list)
    std2 = np.std(numb2_list)
    
    feature1 = std/u
    feature2 = std2/u2
    feature3 = u
    feature4 = len(numb2_list) / float(len(numb_list))
    print('不去重:%s，去重:%s，平均长度:%s，有效聚类数量个数:%s，聚类名称:%s ，内组距为%s' %(feature1,feature2,feature3,feature4,i,cluster_distance))
    if feature1 < 0.2 and feature2 < 0.2 and feature3 > 10 and feature4 > 0.1 and len(numb2_list) > 3:
        res_list += part_res
        print('choose ' + i)
with open ('./h_train.txt','w') as f1:
    for line in res_list:
        f1.write(line)
