"""
This file contains a script for learning encoding-decoding network
on our dataset.

Usage example: python learn_dataset_encoding.py data_dir

Developed by Taras Kucherenko (tarask@kth.se)
"""

import sys
sys.path.append('.')
import numpy as np
import os

import train as autoencoder_training
from utils.utils import prepare_motion_data, DataSet, DataSets

from config import args


def check_params():
    # Check if the dataset exists
    if not os.path.isdir(os.path.abspath(args.data_dir)):
        raise ValueError(f'Path to the dataset ({os.path.abspath(args.data_dir)}) does not exist!\n' + \
                          'Please provide the correct path.')

    # Check if the flags were set properly
    if not os.path.isdir(os.path.abspath(args.chkpt_dir)):
         raise ValueError(f'Path to the checkpoints ({args.chkpt_dir}) does not exist!\n' + \
                           'Please provide the correct path.')

if __name__ == '__main__':
    # Check parameters
    check_params()

    train_normalized_data, train_data, dev_normalized_data, \
    max_val, mean_pose = prepare_motion_data(args.data_dir)

    chkpt_path = os.path.abspath(args.chkpt_dir)

    # Train or load the AE network
    nn = autoencoder_training.learning()

    """       Create save directory for the encoded data         """

    if args.middle_layer == 1:
        save_dir = os.path.join(args.data_dir, str(args.layer1_width))
    elif args.middle_layer == 2:
        save_dir = os.path.join(args.data_dir, str(args.layer2_width))
    else:
        raise("\nMiddle layer is more than 2! change args.middle_layer value\n")

    if not os.path.isdir(save_dir):
        print(f"Created directory {os.path.abspath(save_dir)} for saving the encoded data.")
        os.makedirs(save_dir)


    """                  Encode the train data                   """
    # Encode it
    encoded_train_data = autoencoder_training.encode(nn, train_normalized_data)
    print("\nEncoded train set shape: ", encoded_train_data.shape)
    # And save into file
    np.save(os.path.join(save_dir, "Y_train_encoded.npy"), encoded_train_data)

    """                  Encode the dev data                     """

    # Encode it
    encoded_dev_data = autoencoder_training.encode(nn, dev_normalized_data)
    # And save into files
    np.save(os.path.join(save_dir, "Y_dev_encoded.npy"), encoded_dev_data)
