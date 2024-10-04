import os
import shutil

# 定义normal_models文件夹的路径
normal_models_path = 'normal_models_1'

# 遍历normal_models文件夹中的所有子文件夹
for folder_name in os.listdir(normal_models_path):
    folder_path = os.path.join(normal_models_path, folder_name)
    
    # 检查是否是文件夹
    if os.path.isdir(folder_path):
        # 检查exec_time.json文件是否存在
        exec_time_path = os.path.join(folder_path, 'exec_time.json')
        if not os.path.exists(exec_time_path):
            # 删除model_i文件夹
            shutil.rmtree(folder_path)
            print(f'Deleted {folder_name}')

