"""
This file contains a usage script, intended to test using interface.
Developed by Taras Kucherenko (tarask@kth.se)
"""
import glob
from os import path
import sys
sys.path.append('.')
import numpy as np

import train as tr
from learn_ae_n_encode_dataset import create_nn, prepare_motion_data
# from data_processing import bvh2features
from config import args

import numpy as np

import sys

from scipy.signal import savgol_filter


def smoothing(motion):
    smoothed = [savgol_filter(motion[:, i], 13, 3) for i in range(motion.shape[1])]

    new_motion = np.array(smoothed).transpose()

    return new_motion


if __name__ == '__main__':
    # Make sure that the two mandatory arguments are provided
    if args.input_file is None or args.output_file is None:
        print("Usage: python decode.py -input_dir INPUT_DIR -output_dir OUTPUT_DIR \n" + \
              "Where INPUT_FILE is the encoded prediction file and OUTPUT_FILE is the file in which the decoded gestures will be saved.")
        exit(-1)

    # For decoding these arguments are always False and True
    args.pretrain_network = False
    args.load_model_from_checkpoint = True

    # Get the data
    Y_train_normalized, Y_train, Y_dev_normalized, max_val, mean_pose = prepare_motion_data(args.data_dir)

    # Train the network
    nn = create_nn(Y_train_normalized, Y_dev_normalized, max_val, mean_pose)

    input_files = []
    input_files = sorted([f for f in glob.glob(args.input_file + '/*.npy')])

    # idx = 0
    for f in input_files:
        # Read the encoding
        encoding = np.load(f)

        print(encoding.shape)

        # Decode it
        decoding = tr.decode(nn, encoding)

        # Smoothing
        decoding = smoothing(decoding)

        print(decoding.shape)

        name = path.basename(f)
        output_path = path.join(args.output_file, f'decoded_{name}')
        np.save(output_path, decoding)
        # idx += 1

        # bvh2features.feat2bvh(args.output_file, decoding, name)


    # Close Tf session
    nn.session.close()



# """
# This file contains a usage script, intended to test using interface.
# Developed by Taras Kucherenko (tarask@kth.se)
# """
# import sys
# sys.path.append('.')
# import numpy as np
#
# import train as tr
# from learn_ae_n_encode_dataset import create_nn, prepare_motion_data
# from config import args
#
# import numpy as np
#
# import sys
#
# from scipy.signal import savgol_filter
#
#
# def smoothing(motion):
#
#     smoothed = [savgol_filter(motion[:,i], 13, 3) for i in range(motion.shape[1])]
#
#     new_motion = np.array(smoothed).transpose()
#
#     return new_motion
#
#
# if __name__ == '__main__':
#     # Make sure that the two mandatory arguments are provided
#     if args.input_file is None or args.output_file is None:
#         print("Usage: python decode.py -input_file INPUT_FILE -output_file OUTPUT_FILE \n" + \
#               "Where INPUT_FILE is the encoded prediction file and OUTPUT_FILE is the file in which the decoded gestures will be saved.")
#         exit(-1)
#
#     # For decoding these arguments are always False and True
#     args.pretrain_network = False
#     args.load_model_from_checkpoint = True
#
#     # Get the data
#     Y_train_normalized, Y_train, Y_dev_normalized, max_val, mean_pose  = prepare_motion_data(args.data_dir)
#
#     # Train the network
#     nn = create_nn(Y_train_normalized, Y_dev_normalized, max_val, mean_pose)
#
#     # Read the encoding
#     encoding = np.load(args.input_file)
#
#     print(encoding.shape)
#
#     # Decode it
#     decoding = tr.decode(nn, encoding)
#
#     # Smoothing
#     decoding = smoothing(decoding)
#
#     print(decoding.shape)
#
#     np.save(args.output_file, decoding)
#
#     # Close Tf session
#     nn.session.close()
