# 要对已经经过error_classify.py处理过的模型再进行筛选，error_classify.py会筛选err.log中没有精度错误的模型
import os
import shutil

# 定义源文件夹和目标文件夹的路径
source_folder = "error_other_1"
target_folder = "error_op_1"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 获取source_folder中所有的子文件夹
for model_folder_name in os.listdir(source_folder):
    model_folder_path = os.path.join(source_folder, model_folder_name)
    # 检查是否为文件夹
    if os.path.isdir(model_folder_path):
        bug_report_folder = os.path.join(model_folder_path, "bug_report")
        err_log_path = os.path.join(bug_report_folder, "err.log")
        # 检查err.log文件是否存在
        if os.path.isfile(err_log_path):
            with open(err_log_path, 'r') as file:
                contents = file.read()
                # 检查文件内容是否包含指定字符串
                if "Unsupported operator" in contents or "unknown intrinsic Op" in contents:
                    # 如果包含，移动model_i文件夹到error_op中
                    shutil.move(model_folder_path, os.path.join(target_folder, model_folder_name))

# 打印完成消息
print("Files have been sorted.")

