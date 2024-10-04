import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np
from typing import List
import re
import numpy as np
import os
import json
import time
import shutil

import onnx
import tvm
from tvm.relay.frontend import onnx as ox
from tvm import relay
from tvm.autotvm.graph_tuner.utils import expr2graph

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch import nn

def extract_tensor_info(output_str):
    pattern = r'Tensor(?:Type)?\(\[(.*?)\], (\w+)\)|Tensor\[\((.*?)\), (\w+)\]'
    matches = re.findall(pattern, output_str)

    extracted_info = []
    for match in matches:
        numbers, dtype = match[0] if match[0] else match[2], match[1] if match[1] else match[3]
        numbers_clean = re.sub(r'T\.int64\((\d+)\)', r'\1', numbers)
        numbers_clean = re.sub(r'\s+', '', numbers_clean)
        extracted_info.append((numbers_clean, dtype))

    return extracted_info

def remove_suffix(s):
    return re.sub(r'(i64|i32|f64|f32)$', '', s)

dataset = []
models_path = 'onnx_models_gen_tvm-v0.11.1'
transform_error_dir = './transform_error'
# 创建 transform_error 文件夹（如果不存在）
if not os.path.exists(transform_error_dir):
    os.makedirs(transform_error_dir)

models_count = 0
for sub_dir in os.listdir(models_path):
    if sub_dir.startswith("model_") and sub_dir[6:].isdigit():
        models_count += 1
print('The count of models: ',models_count)

processed_models = 0  # Count of successfully processed models
total_time = 0  # Total time taken to process models

for iteration_count in range(models_count):
    model_start_time = time.time()  # Start timer for this model

    if iteration_count % 50 == 0:
        print('iteration: ', iteration_count)
    model_path_dic = "onnx_models_gen_tvm-v0.11.1/model_{}".format(iteration_count)
    model_path = "{}/model.onnx".format(model_path_dic)
    try:
        onnx_model = onnx.load(model_path)
    except Exception as e:
        # print(f"加载模型 {iteration_count + 1} 时出错：{e}")
        shutil.move(model_path_dic, transform_error_dir)
        continue

    try:
        relay_ir = ox.from_onnx(onnx_model)
    except Exception as e:
        # print(f"relay转换onnx模型 {iteration_count + 1} 时出错: {e}")
        shutil.move(model_path_dic, transform_error_dir)
        continue  # 跳过此模型，进入下一次迭代

    tvm_model = relay_ir[0]
    func = tvm_model.__getitem__('main')
    expr = func
    node_dict = {}
    node_list = []
    target_ops = []
    tvm_target = tvm.target.Target('llvm')
    expr2graph(expr, target_ops, node_dict, node_list, tvm_target) 

    for node in node_list:
        input_copy = []
        for input in node['inputs']:
            input_copy.append(input[0])
        node['inputs'] = input_copy    
    count_node = 0
    for node in node_list:
        node['id'] = count_node
        count_node = count_node + 1
    # 增加每个节点的输入张量信息
    for node in node_list:
        node_id = node['id']
        node_op = node['op']
        node_inputs = node['inputs']
        node_types = node['types']
        # 提取每个节点的输入节点的 types，并添加到 input_tensors
        input_tensors = []
        for input_id in node_inputs:
            for input_node_info in node_list:
                if input_node_info['id'] == input_id:
                    input_tensors.extend(input_node_info['types'])
                    break
        node['input_tensors'] = input_tensors
    for node in node_list:
        if type(node['node']) == tvm.relay.expr.Var:
            node['category'] = 'Var'
        elif type(node['node']) == tvm.relay.expr.Constant:
            node['category'] = 'Constant'
        elif type(node['node']) == tvm.relay.expr.Tuple:
            node['category'] = 'Tuple'
        elif type(node['node']) == tvm.relay.expr.Call:
            node['category'] = 'Call'
        else:
            node['category'] = 'unknown'
    edge_info = []
    for node in node_list:
        node_id_to = node['id']
        for node_id_from in node['inputs']:
            edge_info.append([node_id_from, node_id_to, str(node_list[node_id_from]['types'][0])])      
    for edge in edge_info:
        output_str = str(edge[2])
        del edge[2]
        shape = [0, 0, 0, 0, 0]
        dtype_str = 'unknown'
        matches = extract_tensor_info(output_str)
        if len(matches) > 0:
            for match in matches:
                shape_str, dtype_str = match
                if shape_str.isdigit():
                    shape = [int(shape_str)]
                elif len(shape_str) > 2:
                    shape = [int(remove_suffix(dim)) for dim in shape_str.strip('[]').split(',')]
                else:
                    shape = []
                shape += [0] * (5 - len(shape))
        edge.append(shape)
        edge.append(dtype_str)   
    for index_edge in range(len(edge_info)):
        edge_info[index_edge].append(index_edge)
    node_info = []
    for node in node_list:
        op_name = str(node['op'])
        if op_name == 'None':  # 如果结果是字符串 'None'
            op_name = 'None'
        elif 'Op(' in op_name and op_name.endswith(')'):  # 如果是形如 'Op(...)'
            op_name = op_name
        elif not op_name.startswith('Op(') or not op_name.endswith(')'):  # 不是 'None' 也不是 'Op(...)'
            op_name = f'Op({op_name})'

        node_info.append([node['id'], node['inputs'], op_name, node['category']])


    bug_report_path = os.path.join(model_path_dic, "bug_report")
    bug_or_no = 0
    if os.path.exists(bug_report_path):
        bug_or_no = 1
    graph_info = []
    graph_info.append(len(node_info))
    graph_info.append(len(edge_info))
    op_list = []
    category_list = []
    for node in node_info:
        op_list.append(node[2])
        category_list.append(node[3])
    op_set = set(op_list)
    category_set = set(category_list)
    graph_info.append(list(op_set))
    graph_info.append(list(category_set))
    dataset.append({
        'node_info': node_info,
        'edge_info': edge_info,
        'graph_info': graph_info,
        'result': bug_or_no
    })


    model_end_time = time.time()  # End timer for this model
    model_time = model_end_time - model_start_time
    total_time += model_time
    processed_models += 1  # Increment count of successfully processed models
    average_time_per_model = total_time / processed_models if processed_models > 0 else 0


    if iteration_count % 50 == 0:
        print(f"Model {iteration_count} processed in {model_time:.2f} seconds")
        print(f"Average time per model so far: {average_time_per_model:.2f} seconds")


print(f"Average time per model so far: {average_time_per_model:.2f} seconds")
print(f"Successfully passed model count: {processed_models:.2f} ")

json_filename = 'onnx_models_gen_tvm-v0.11.1.json'
with open(json_filename, 'w') as jsonfile:
    json.dump(dataset, jsonfile, indent=4)

    
    
    
    
    
    
    
    
    
    
    
