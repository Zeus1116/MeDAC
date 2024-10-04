import os
import shutil

# 定义源文件夹路径和目标文件夹路径
source_folder = "normal_models_5"
error_folder = "error_accuracy_5"

models_path = 'normal_models_5'

models_count = 0
for sub_dir in os.listdir(models_path):
    if sub_dir.startswith("model_") and sub_dir[6:].isdigit():
        models_count += 1
print('The count of models: ',models_count)

# 创建目标文件夹
os.makedirs(error_folder, exist_ok=True)

# 遍历源文件夹中的子文件夹
for i in range(models_count):
    model_folder = os.path.join(source_folder, f"model_{i}")
    # 检查子文件夹是否存在以及是否包含 bug_report 子文件夹
    if os.path.isdir(model_folder):
        bug_report_folder = os.path.join(model_folder, "bug_report")
        if os.path.isdir(bug_report_folder):
            # 移动含有 bug_report 的子文件夹到目标文件夹中
            shutil.move(model_folder, error_folder)

