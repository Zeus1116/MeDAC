#!/bin/bash

# 创建models目录
mkdir -p onnx_models_gen_tvm-v0.11.1

# 循环生成10000个模型
for i in {0..299999}
do
    echo "Running iteration $i"
    # 生成1到100之间的随机整数
    MAX_NODES=$((1 + $RANDOM % 30))

    # 生成模型
    nnsmith.model_gen model.type=onnx debug.viz=true backend.type=tvm mgen.max_nodes=$MAX_NODES
    
    # 移动并重命名nnsmith_output到models/model_i
    mv nnsmith_output "onnx_models_gen_tvm-v0.11.1/model_$i"

done

