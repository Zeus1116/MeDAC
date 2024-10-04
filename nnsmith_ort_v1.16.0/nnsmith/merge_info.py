import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
from typing import List
import re
import os
import json
import time
import shutil
import random

data_info_list_total = []
time_info_list_total = []
for iteration_count_hhh in range(1):
    data_info_list = []
    time_info_list = []
    models_path = 'normal_models'
    models_count = 0
    for sub_dir in os.listdir(models_path):
        if sub_dir.startswith("model_") and sub_dir[6:].isdigit():
            models_count += 1
    print('The count of normal models: ',models_count)
    normal_random_elements = random.sample(range(models_count), 100000)
    for iteration_count in normal_random_elements:
        model_path_dic = "normal_models/model_{}".format(iteration_count)
        model_path = "{}/model.onnx".format(model_path_dic)
        data_info_json_filename = '{}/data_info.json'.format(model_path_dic)
        with open(data_info_json_filename, 'r') as jsonfile:
            data_info = json.load(jsonfile)
            data_info = data_info[0]
            data_info['result']=0
        data_info_list.append(data_info)
        
        gen_time_info_json_filename = '{}/gen_time.json'.format(model_path_dic)
        with open(gen_time_info_json_filename, 'r') as jsonfile:
            gen_time_info = json.load(jsonfile)
            gen_time_info = gen_time_info[0]

        exec_time_info_json_filename = '{}/exec_time.json'.format(model_path_dic)
        with open(exec_time_info_json_filename, 'r') as jsonfile:
            exec_time_info = json.load(jsonfile)
            exec_time_info = exec_time_info[0]
        time_info_list.append({
            'tgen':gen_time_info['tgen'],
            'tmat':gen_time_info['tmat'],
            'tfetch':gen_time_info['tfetch'],
            'tsave':gen_time_info['tsave'],
            'texec':exec_time_info['exec_time']
        })

    models_path = 'error_accuracy'
    models_count = 0
    for sub_dir in os.listdir(models_path):
        if sub_dir.startswith("model_") and sub_dir[6:].isdigit():
            models_count += 1
    print('The count of error accuracy models: ',models_count)
    error_accuracy_random_elements = random.sample(range(models_count), 4)
    for iteration_count in range(0):
        model_path_dic = "error_accuracy/model_{}".format(iteration_count)
        model_path = "{}/model.onnx".format(model_path_dic)
        data_info_json_filename = '{}/data_info.json'.format(model_path_dic)
        with open(data_info_json_filename, 'r') as jsonfile:
            data_info = json.load(jsonfile)
            data_info = data_info[0]
            data_info['result']=1
        data_info_list.append(data_info)
        
        gen_time_info_json_filename = '{}/gen_time.json'.format(model_path_dic)
        with open(gen_time_info_json_filename, 'r') as jsonfile:
            gen_time_info = json.load(jsonfile)
            gen_time_info = gen_time_info[0]

        exec_time_info_json_filename = '{}/exec_time.json'.format(model_path_dic)
        with open(exec_time_info_json_filename, 'r') as jsonfile:
            exec_time_info = json.load(jsonfile)
            exec_time_info = exec_time_info[0]
        time_info_list.append({
            'tgen':gen_time_info['tgen'],
            'tmat':gen_time_info['tmat'],
            'tfetch':gen_time_info['tfetch'],
            'tsave':gen_time_info['tsave'],
            'texec':exec_time_info['exec_time']
        })

    models_path = 'error_other_only'
    models_count = 0
    for sub_dir in os.listdir(models_path):
        if sub_dir.startswith("model_") and sub_dir[6:].isdigit():
            models_count += 1
    print('The count of error other models: ',models_count)
    error_other_random_elements = random.sample(range(models_count), 9)
    for iteration_count in error_other_random_elements:
        model_path_dic = "error_other_only/model_{}".format(iteration_count)
        model_path = "{}/model.onnx".format(model_path_dic)
        data_info_json_filename = '{}/data_info.json'.format(model_path_dic)
        with open(data_info_json_filename, 'r') as jsonfile:
            data_info = json.load(jsonfile)
            data_info = data_info[0]
            data_info['result']=1
        data_info_list.append(data_info)
        
        gen_time_info_json_filename = '{}/gen_time.json'.format(model_path_dic)
        with open(gen_time_info_json_filename, 'r') as jsonfile:
            gen_time_info = json.load(jsonfile)
            gen_time_info = gen_time_info[0]

        exec_time_info_json_filename = '{}/exec_time.json'.format(model_path_dic)
        with open(exec_time_info_json_filename, 'r') as jsonfile:
            exec_time_info = json.load(jsonfile)
            exec_time_info = exec_time_info[0]
        time_info_list.append({
            'tgen':gen_time_info['tgen'],
            'tmat':gen_time_info['tmat'],
            'tfetch':gen_time_info['tfetch'],
            'tsave':gen_time_info['tsave'],
            'texec':exec_time_info['exec_time']
        })
        # 将data_info_list和time_info_list合并
    combined_list = list(zip(data_info_list, time_info_list))

    # 使用 random.shuffle() 函数打乱列表
    random.shuffle(combined_list)

    # 打印打乱后的列表
    # print(combined_list)

    # 使用 zip() 函数将打乱后的列表分开
    shuffled_data_info_list, shuffled_time_info_list = zip(*combined_list)

    json_filename = './ort-v1.16.0-testdata_0_9/demo_test_data_info_{}.json'.format(iteration_count_hhh)
    with open(json_filename, 'w') as jsonfile:
        json.dump(shuffled_data_info_list, jsonfile, indent=4)
    
    json_filename = './ort-v1.16.0-testdata_0_9/demo_test_time_info_{}.json'.format(iteration_count_hhh)
    with open(json_filename, 'w') as jsonfile:
        json.dump(shuffled_time_info_list, jsonfile, indent=4)


# op_set = {'None'}
# data_type_set = {'Unknow'}
# op_names = []
# data_type_names = []
# for data_info in data_info_list:
#     node_info = data_info['node_info']
#     for node in node_info:
#         op_set.add(node[1])
#     edge_info = data_info['edge_info']
#     for edge in edge_info:
#         data_type_set.add(edge[3])
# op_names = list(op_set)
# data_type_names = list(data_type_set)
# output = ", ".join([f'"{name}"' for name in data_type_set])
# print(output)
# output = ", ".join([f'"{name}"' for name in op_set])
# print(output)

## 将data_info_list和time_info_list合并
# combined_list = list(zip(data_info_list, time_info_list))

# # 使用 random.shuffle() 函数打乱列表
# random.shuffle(combined_list)

# # 打印打乱后的列表
# # print(combined_list)

# # 使用 zip() 函数将打乱后的列表分开
# shuffled_data_info_list, shuffled_time_info_list = zip(*combined_list)

# json_filename = 'demo_test_data_info.json'
# with open(json_filename, 'w') as jsonfile:
#     json.dump(shuffled_data_info_list, jsonfile, indent=4)
  
# json_filename = 'demo_test_time_info.json'
# with open(json_filename, 'w') as jsonfile:
#     json.dump(shuffled_time_info_list, jsonfile, indent=4)




