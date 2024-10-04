
import os
import shutil
import re

# 定义文件夹路径
error_other_path = './error_other_final'
error_other_only_path = './error_other_only_2'

# 如果error_other_only文件夹不存在，则创建它
if not os.path.exists(error_other_only_path):
    os.makedirs(error_other_only_path)

# 用于记录已经遇到过的错误标识符的集合
seen_errors = set()

# 用于提取Traceback中以"0:"开始，"<"或"("结束的行内容的正则表达式
# 确保"0:"是行的开头，可以有空格，但不能有其他字符
stack_trace_last_line_pattern = re.compile(r'\b[01]: (.*?)[<(]')

# 遍历error_other文件夹中的所有model_i子文件夹
for model_folder in os.listdir(error_other_path):
    model_path = os.path.join(error_other_path, model_folder)
    bug_report_path = os.path.join(model_path, 'bug_report')
    err_log_path = os.path.join(bug_report_path, 'err.log')

    if os.path.isdir(model_path) and os.path.exists(err_log_path):
        # 读取err.log文件内容
        with open(err_log_path, 'r') as file:
            err_log_content = file.read()

        # 尝试提取Traceback中符合条件的最后一个行内容
        last_line_match = stack_trace_last_line_pattern.search(err_log_content)
        if last_line_match:
            # 获取匹配项，并替换掉空格
            error_function_signature = last_line_match.group(1).replace(' ', '')

            # 将处理后的错误标签添加到seen_errors集合中
            if error_function_signature not in seen_errors:
                seen_errors.add(error_function_signature)
                print(f'Added new error signature: {error_function_signature}')

                # 将model_i文件夹复制到error_other_only中
                destination_path = os.path.join(error_other_only_path, model_folder)
                shutil.copytree(model_path, destination_path)
                print(f'Copied {model_folder} to error_other_only')

print('Task completed.')


