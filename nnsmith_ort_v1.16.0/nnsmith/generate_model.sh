#!/bin/bash

# 创建models目录
mkdir -p normal_models_2

# 循环生成10000个模型
for i in {0..249999}
do
    echo "Running iteration $i"
    # 生成1到30之间的随机整数
    MAX_NODES=$((1 + $RANDOM % 30))

    # 生成模型
    nnsmith.model_gen model.type=onnx debug.viz=true backend.type=onnxruntime mgen.max_nodes=$MAX_NODES
    
    # 移动并重命名nnsmith_output到models/model_i
    mv nnsmith_output "normal_models_2/model_$i"

done

