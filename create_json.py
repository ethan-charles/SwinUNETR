import os
import json

def create_json(directory, output_file):
    data = {"training": []}
    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    num_directories = len(subdirectories)
    fold_0_limit = int(0.7 * num_directories)  # 70% of the directories

    # Iterate over each subdirectory in the main directory
    for idx, subdir in enumerate(subdirectories):
        fold_number = 0 if idx < fold_0_limit else 1
        entry = {
            "fold": fold_number,
            "image": [],
            "label": ""
        }
        
        # List all files and filter based on file extensions
        for file in os.listdir(subdir):
            filepath = os.path.join(subdir, file)
            if file.endswith(".nii.gz"):
                if "seg" in file:
                    entry["label"] = filepath
                else:
                    entry["image"].append(filepath)
        
        # Ensure we capture only directories that have both images and label
        if entry["image"] and entry["label"]:
            data["training"].append(entry)

    # Write the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Usage
directory = '/root/dataset_val/MICCAI_BraTS2020_TrainingData'  # Your main directory containing subdirectories
output_file = '/root/swinUNETR/jsons/dataset_validation.json'  # Name of the output JSON file
create_json(directory, output_file)
