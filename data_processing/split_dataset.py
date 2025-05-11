"""
This script is used to split the dataset into train, test and dev sets.
More info on its usage is given in the main README.md file

@authors: Taras Kucherenko, Rajmund Nagy
"""
import glob
import sys
import os
import shutil
import pandas
from os import path
import random

# Params
from data_params import parser


audio_prefix = 'music_'
motion_prefix = 'motion_'

def copy_files(idx, raw_d_dir, processed_d_dir, data_split, suffix=""):
    # Copy audio
    audio_filename = f'{audio_prefix}{idx}.wav'
    audio_origin_files = sorted([f for f in glob.glob(raw_d_dir + '/wav/*.wav')])

    target_file_path = path.join(processed_d_dir, data_split, "inputs", audio_filename)
    shutil.copy(audio_origin_files[int(idx)], target_file_path)

    # Copy gestures
    motion_filename = f'{motion_prefix}{idx}.npz'
    motion_origin_files = sorted([f for f in glob.glob(raw_d_dir + '/bvh/*/*.npz')])

    target_file_path = path.join(processed_d_dir, data_split, "labels", motion_filename)
    shutil.copy(motion_origin_files[int(idx)], target_file_path)


def create_dataset_splits(raw_d_dir, processed_d_dir):
    """Create the train/dev/test splits in new subfolders within 'processed_d_dir'."""
    _create_data_directories(processed_d_dir)

    data_list = os.listdir(path.join(raw_d_dir, 'wav'))

    all_data_num = len(data_list)
    val_num = int(all_data_num * 0.1)

    random_val_idx = []
    random_val_idx = random.sample(range(0, all_data_num - 1), val_num)
    random_val_idx.sort()
    print('val#: ', random_val_idx)

    for i in range(0, all_data_num):
        # prepare training data
        if not i in random_val_idx:
            copy_files(str(i).zfill(3), raw_d_dir, processed_d_dir, "train")
        # prepare dev data
        else:
            copy_files(str(i).zfill(3), raw_d_dir, processed_d_dir, "val")


    extracted_dir = path.join(processed_d_dir)

    val_files, train_files = _format_datasets(extracted_dir, random_val_idx, all_data_num)

    # Save the filenames of each datapoints (the preprocessing script will use these)
    val_files.to_csv(path.join(extracted_dir, "dev-dataset-info.csv"), index=False)
    train_files.to_csv(path.join(extracted_dir, "train-dataset-info.csv"), index=False)


def _create_data_directories(processed_d_dir):
    """Create subdirectories for the dataset splits."""
    dir_names = ["val", "train"]
    sub_dir_names = ["inputs", "labels"]
    os.makedirs(processed_d_dir, exist_ok=True)

    print("Creating the datasets in the following directories:")
    for dir_name in dir_names:
        dir_path = path.join(processed_d_dir, dir_name)
        print('  ', path.abspath(dir_path))
        os.makedirs(dir_path, exist_ok=True)  # e.g. ../../dataset/processed/train

        for sub_dir_name in sub_dir_names:
            dir_path = path.join(processed_d_dir, dir_name, sub_dir_name)
            os.makedirs(dir_path, exist_ok=True)  # e.g. ../../dataset/processed/train/inputs/
    print()


def _format_datasets(extracted_dir, val_num, all_data_num):
    print("The datasets will contain the following indices:", end='')
    train_idx = []
    for i in range(0, all_data_num):
        if not i in val_num:
            train_idx.append(i)

    val_files = _files_to_pandas_dataframe(extracted_dir, "val", val_num)
    train_files = _files_to_pandas_dataframe(extracted_dir, "train", train_idx)
    print()

    return val_files, train_files


def _files_to_pandas_dataframe(extracted_dir, set_name, idx_range):
    info_msg = f"\n  {set_name}:"
    print("{:10}".format(info_msg), end='')

    files = []
    for idx in idx_range:
        # original files
        input_file = path.abspath(
            path.join(extracted_dir, set_name, "inputs", audio_prefix + str(idx).zfill(3) + ".wav"))
        label_file = path.abspath(
            path.join(extracted_dir, set_name, "labels", motion_prefix + str(idx).zfill(3) + ".npz"))
        if os.path.isfile(input_file):
            files.append((input_file, label_file))

            print(idx, end=' ')

    return pandas.DataFrame(data=files, columns=["wav_filename", "bvh_filename"])


def check_dataset_directories(raw_data_dir):

    if not path.isdir(raw_data_dir):
        abs_path = path.abspath(raw_data_dir)

        print(f"ERROR: The given dataset folder for the raw data ({abs_path}) does not exist!")
        print("Please, provide the correct path to the dataset in the `-raw_data_dir` argument.")
        exit(-1)

    music_dir = path.join(raw_data_dir, 'wav')
    motion_dir = path.join(raw_data_dir, 'bvh')

    for sub_dir in [music_dir, motion_dir]:
        if not path.isdir(sub_dir):
            _, name = path.split(sub_dir)
            print(f"ERROR: The '{name}' directory is missing from the given dataset folder: '{raw_data_dir}'!")
            exit(-1)


if __name__ == "__main__":
    args = parser.parse_args()

    check_dataset_directories(args.raw_data_dir)
    create_dataset_splits(args.raw_data_dir, args.proc_data_dir)

    print(f"\nSplit data finished!")


