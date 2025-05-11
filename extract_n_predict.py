"""
This script generates gestures output based on the speech input.
The gestures will be written in the text file:
3d coordinates together with the velocities.
"""

import glob
from os import path

import sys
from tensorflow.keras.models import load_model
import numpy as np

from data_processing.data_params import parser
from data_processing.process_dataset import encode_music_vectors_no_bvh as enc_music
from data_processing.process_dataset_26 import _encode_vectors_no_bvh as enc_audio

def audio_to_feature(model_file, input_dir, output_dir):

    music_files = sorted([f for f in glob.glob(input_dir + '/*.wav')])
    print('music_files: ', music_files)

    for file in music_files:
        input_vector = enc_audio(file, mode='tst', args=args, augment_with_context=True)

        file_name = path.basename(file)
        file_name = file_name.split('.')[0]
        print('filename: ', file_name)

        save_path = path.join(input_dir, f"X_{file_name}.npy")
        np.save(save_path, input_vector)
        re_predict(model_file, input_vector, output_dir, file_name)


def music_to_feature(model_file, input_dir, output_dir):

    music_files = sorted([f for f in glob.glob(input_dir + '/*.wav')])
    print('music_files: ', music_files)

    for file in music_files:
        input_vector = enc_music(file, mode='tst', args=args, augment_with_context=True)

        file_name = path.basename(file)
        file_name = file_name.split('.')[0]
        print('filename: ', file_name)

        # save_path = path.join(input_dir, f"X_{file_name}.npy")
        # np.save(save_path, input_vector)
        re_predict(model_file, input_vector, output_dir, file_name)



def re_predict(model_name, input_file, output_dir, filename):

    model = load_model(model_name)

    # audio = np.load(input_file)
    predicted = np.array(model.predict(input_file))

    print("Predicted shape is: ", predicted.shape)

    save_path = path.join(output_dir, f'predicted_{filename}.npy')
    np.save(save_path, predicted)


# def predict(model_name, input_dir, output_dir):
#     """ Predict human gesture based on the speech
#
#     Args:
#         model_name:  name of the Keras model to be used
#         input_file:  file name of the audio input
#         output_file: file name for the gesture output
#
#     Returns:
#
#     """
#     input_files = []
#     input_files = sorted([f for f in glob.glob(input_dir + '/*.npy')])
#
#     model = load_model(model_name)
#
#     for f in input_files:
#         print(f)
#         audio = np.load(f)
#         predicted = np.array(model.predict(audio))
#
#         print("Encoding shape is: ", predicted.shape)
#
#         name = f.split('/')[-1]
#         output_file = path.join(output_dir, f'predict_{name[-7:]}')
#
#         np.save(output_file, predicted)


if __name__ == "__main__":
    args = parser.parse_args()

    # # Check if script get enough parameters
    # if len(sys.argv) < 4:
    #     raise ValueError('Not enough paramters! \nUsage : python ' + sys.argv[0].split("/")[-1] +
    #                      ' MODEL_NAME INPUT_FILE OUTPUT_FILE')

    feature_n = 26

    model_file = args.model_name
    if not model_file.endswith(".hdf5"):
        model_file += ".hdf5"

    input_dir = args.raw_data_dir
    output_dir = args.proc_data_dir

    if (feature_n == 36):
        music_to_feature(model_file, input_dir, output_dir)
    elif (feature_n == 26):
        audio_to_feature(model_file, input_dir, output_dir)

    else:
        print('Value Error!!')

    # predict(model_file, sys.argv[2], sys.argv[3])
    print("Done")



# """
# This script generates gestures output based on the speech input.
# The gestures will be written in the text file:
# 3d coordinates together with the velocities.
# """
#
# import sys
# from keras.models import load_model
# import numpy as np
#
# def predict(model_name, input_file, output_file):
#     """ Predict human gesture based on the speech
#
#     Args:
#         model_name:  name of the Keras model to be used
#         input_file:  file name of the audio input
#         output_file: file name for the gesture output
#
#     Returns:
#
#     """
#     model = load_model(model_name)
#     audio = np.load(input_file)
#
#     predicted = np.array(model.predict(audio))
#
#     print("Encoding shape is: ", predicted.shape)
#     np.save(output_file, predicted)
#
#
# if __name__ == "__main__":
#
#     # Check if script get enough parameters
#     if len(sys.argv) < 4:
#         raise ValueError('Not enough paramters! \nUsage : python ' + sys.argv[0].split("/")[-1] +
#                          ' MODEL_NAME INPUT_FILE OUTPUT_FILE')
#
#     model_file = sys.argv[1]
#     if not model_file.endswith(".hdf5"):
#         model_file += ".hdf5"
#
#     predict(model_file, sys.argv[2], sys.argv[3])
#     print("Done")
