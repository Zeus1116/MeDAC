import os
import shutil
import re

# 定义文件夹路径
error_other_path = './error_other'
error_other_only_path = './error_other_only'

# 如果error_other_only文件夹不存在，则创建它
if not os.path.exists(error_other_only_path):
    os.makedirs(error_other_only_path)

# 用于记录已经遇到过的错误标识符的集合
seen_errors = set()

# 用于提取Traceback中最后一行开始到第一个":"结束的正则表达式
stack_trace_last_part_pattern = re.compile(r'^(.*?)(?=:)', re.MULTILINE)

# 遍历error_other文件夹中的所有model_i子文件夹
for model_folder in os.listdir(error_other_path):
    model_path = os.path.join(error_other_path, model_folder)
    bug_report_path = os.path.join(model_path, 'bug_report')
    err_log_path = os.path.join(bug_report_path, 'err.log')

    if os.path.isdir(model_path) and os.path.exists(err_log_path):
        # 读取err.log文件内容
        with open(err_log_path, 'r') as file:
            err_log_content = file.read()

        # 尝试提取Traceback中最后一行开始到第一个":"结束的部分
        last_part_matches = stack_trace_last_part_pattern.findall(err_log_content)
        if last_part_matches:
            # 获取所有匹配项的最后一项，即最后一行的错误标识符
            error_function_signature = last_part_matches[-1].strip()

            # 将处理后的错误标签添加到seen_errors集合中
            if error_function_signature not in seen_errors:
                seen_errors.add(error_function_signature)
                print(f'Added new error signature: {error_function_signature}')

                # 将model_i文件夹复制到error_other_only中
                destination_path = os.path.join(error_other_only_path, model_folder)
                shutil.copytree(model_path, destination_path)
                print(f'Copied {model_folder} to error_other_only')

print('Task completed.')
