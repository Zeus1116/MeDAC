#!/bin/bash

# 获取onnx_models_gen中model_i子文件夹的数量
model_count=$(find onnx_models_gen_tvm-v0.11.1 -type d -name 'model_*' | wc -l)

# 初始化总执行时间
total_exec_time=0

# 循环遍历子文件夹
for i in $(seq 0 $((model_count-1))); do
    echo "Iteration ${i}"
    model_folder="onnx_models_gen_tvm-v0.11.1/model_${i}"
    # 复制子文件夹到当前目录下，并重命名为nnsmith_output
    cp -r "$model_folder" ./nnsmith_output

    # 开始计时
    # start_time=$(date +%s%3N)

    # 运行模型并与oracle中的准确结果做对比
    # 注意替换这里的nnsmith.model_exec和backend.type=tvm为实际的执行命令和参数
    nnsmith.model_exec model.type=onnx backend.type=tvm model.path=nnsmith_output/model.onnx

    # 结束计时
    # end_time=$(date +%s%3N)
    # exec_time=$((end_time - start_time))
    # total_exec_time=$((total_exec_time + exec_time))

    # 记录时间
    # echo "Model ${i} execution time: ${exec_time}ms"

    # 删除原有的model_i文件夹
    rm -rf "$model_folder"
    # 将nnsmith_output文件夹移动到onnx_models_gen中，并重命名为model_i
    mv ./nnsmith_output "$model_folder"
done

# 打印总执行时间
# echo "Total execution time: ${total_exec_time}ms"

