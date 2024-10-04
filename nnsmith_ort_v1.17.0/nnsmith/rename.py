import os

def rename_subfolders(parent_folder):
    # 获取所有子文件夹的名称
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    
    # 对子文件夹进行排序，确保它们按照原始的顺序进行重命名
    subfolders.sort(key=lambda x: int(x.split('_')[1]))

    # 重新命名子文件夹
    for i, folder in enumerate(subfolders):
        new_name = f"model_{i}"
        old_path = os.path.join(parent_folder, folder)
        new_path = os.path.join(parent_folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {folder} to {new_name}")

# 调用函数
rename_subfolders("error_other_only")

