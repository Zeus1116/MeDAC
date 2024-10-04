import os
import shutil

# 设置源文件夹和目标文件夹
source_folder = 'onnx_models_gen_tvm-v0.10.0'
target_folder = 'others'

# 创建目标文件夹，如果它不存在
os.makedirs(target_folder, exist_ok=True)

# 开始处理文件夹
for sub_dir in os.listdir(source_folder):
    if sub_dir.startswith("model_"):
        # 提取模型的编号并检查是否大于19999
        model_number = int(sub_dir.split('_')[-1])
        if model_number > 19999:
            # 构建完整的源路径和目标路径
            source_path = os.path.join(source_folder, sub_dir)
            target_path = os.path.join(target_folder, sub_dir)
            
            # 移动文件夹
            shutil.move(source_path, target_path)
            print(f"Moved: {sub_dir}")

