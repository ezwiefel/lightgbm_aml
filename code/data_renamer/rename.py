import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--file_prefix', type=str)
    args, _ = parser.parse_known_args()
    return args.__dict__

def rename_files(input_path, output_path, file_prefix):
    """Copies files to 'train_0.csv' and increases based on the number of files"""
    
    print("Listing directory")
    files = [file for file in os.listdir(input_path) if file.endswith('.csv')]
    print(files)

    for i , file in enumerate(files):
        new_file_name = os.path.join(output_path,f"{file_prefix}_{i}.csv")
        full_path = os.path.join(input_path, file)
        print(f'Copying {file} --> {new_file_name}')
        os.makedirs(output_path, exist_ok=True)
        shutil.copyfile(full_path, new_file_name)

if __name__ == "__main__":
    args = parse_args()
    rename_files(**args)