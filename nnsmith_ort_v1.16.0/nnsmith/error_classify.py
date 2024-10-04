import os
import shutil

# 定义源文件夹和目标文件夹的路径
source_folder = "error_accuracy_2"
target_folder = "error_other_2"

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
                if "Not equal to tolerance" not in contents:
                    # 如果不包含，移动model_i文件夹到error_other中
                    shutil.move(model_folder_path, os.path.join(target_folder, model_folder_name))

# 打印完成消息
print("Files have been sorted.")

