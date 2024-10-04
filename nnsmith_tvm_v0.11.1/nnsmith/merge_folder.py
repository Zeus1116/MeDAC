import os
import shutil

def merge_folders(source_folders, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Count the number of model folders in the target folder
    model_count = len([name for name in os.listdir(target_folder) if name.startswith("model_")])
    print("model_count: ", model_count)
    # Merge folders from each source folder into the target folder
    for source_folder in source_folders:
        for folder_name in os.listdir(source_folder):
            if folder_name.startswith("model_"):
                model_count += 1
                new_folder_name = f"model_{model_count}"
                source_path = os.path.join(source_folder, folder_name)
                target_path = os.path.join(target_folder, new_folder_name)
                shutil.copytree(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")

if __name__ == "__main__":
    base_folder = "onnx_models_gen_tvm-v0.11.1"
    source_folders = [f"{base_folder}_{i}" for i in range(1, 2)]
    target_folder = base_folder
    merge_folders(source_folders, target_folder)

