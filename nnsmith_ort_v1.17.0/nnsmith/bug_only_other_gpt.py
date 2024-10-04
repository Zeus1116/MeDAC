import os
import shutil
from collections import defaultdict

def read_last_line_of_error_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 获取错误日志的最后一行
    last_line = lines[-1].strip()
    # 删除最后一个冒号及其后面的内容
    if ':' in last_line:
        last_line = last_line.rsplit(':', 1)[0].strip()
    return last_line

def classify_errors(error_logs):
    error_classes = defaultdict(list)
    for model, log in error_logs.items():
        error_classes[log].append(model)
    return error_classes

def main():
    base_dir = './'  # error_other 所在的父目录路径
    error_other_dir = os.path.join(base_dir, 'error_other_3')
    output_dir = os.path.join(base_dir, 'error_other_only_3')
    os.makedirs(output_dir, exist_ok=True)

    error_logs = {}
    
    for model_folder in os.listdir(error_other_dir):
        model_path = os.path.join(error_other_dir, model_folder)
        if os.path.isdir(model_path) and model_folder.startswith("model_"):
            err_log_path = os.path.join(model_path, 'bug_report', 'err.log')
            if os.path.exists(err_log_path):
                error_logs[model_folder] = read_last_line_of_error_log(err_log_path)

    classified_errors = classify_errors(error_logs)
    
    for error_class, models in classified_errors.items():
        if models:
            source_folder = os.path.join(error_other_dir, models[0])
            destination_folder = os.path.join(output_dir, models[0])
            shutil.copytree(source_folder, destination_folder)

if __name__ == "__main__":
    main()

