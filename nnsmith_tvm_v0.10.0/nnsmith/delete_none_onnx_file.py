import os
import shutil

def count_model_folders(root_folder):
    model_folder_count = 0
    while True:
        folder_name = f"model_{model_folder_count}"
        folder_path = os.path.join(root_folder, folder_name)
        if not os.path.exists(folder_path):
            break
        model_folder_count += 1
    return model_folder_count

def remove_folders_without_onnx(root_folder):
    model_folder_count = count_model_folders(root_folder)
    for i in range(model_folder_count):
        folder_name = f"model_{i}"
        folder_path = os.path.join(root_folder, folder_name)
        onnx_file_path = os.path.join(folder_path, "model.onnx")

        # 如果model.onnx文件不存在，则删除文件夹
        if not os.path.exists(onnx_file_path):
            print(f"Deleting folder {folder_name} because model.onnx is not found.")
            # 递归删除文件夹及其内容
            shutil.rmtree(folder_path)

# 指定要检查的文件夹路径
folder_path = "normal_models_1"

# 调用函数进行统计和删除
remove_folders_without_onnx(folder_path)

