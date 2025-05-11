from os import path
import numpy as np

from argparse import ArgumentParser
# import tqdm


def concat_data(data_dir, filename1, filename2, filename3):


    x_data1 = path.join(data_dir, f'X_{filename1}.npy')
    x_data2 = path.join(data_dir, f'X_{filename2}.npy')
    x_data3 = path.join(data_dir, f'X_{filename3}.npy')

    y_data1 = path.join(data_dir, f'Y_{filename1}.npy')
    y_data2 = path.join(data_dir, f'Y_{filename2}.npy')
    y_data3 = path.join(data_dir, f'Y_{filename3}.npy')

    print('Loading...')
    input_vector2 = np.load(x_data2)
    print('Loading...')
    input_vector1 = np.load(x_data1)
    # print('Loading...')
    # input_vector3 = np.load(x_data3)

    # output_vector1 = np.load(y_data1)
    # output_vector2 = np.load(y_data2)
    # output_vector3 = np.load(y_data3)

    print(f'x_data1: {input_vector1.shape}, x_data2: {input_vector2.shape}')
    # print(f'x_data1: {input_vector1.shape}, y_data1: {output_vector1.shape}\n'
    #       f'x_data2: {input_vector2.shape}, y_data2: {output_vector2.shape}\n'
    #       f'x_data3: {input_vector3.shape}, y_data3: {output_vector3.shape}\n')

    X = np.concatenate((input_vector1, input_vector2), axis=0)
    # X = np.concatenate((input_vector1, input_vector2, input_vector3), axis=0)
    # Y = np.concatenate((output_vector1, output_vector2, output_vector3), axis=0)

    del input_vector1, input_vector2
    print('The files were deleted')

    x_save_path = path.join(data_dir, 'X_trn.npy')
    # x_save_path = path.join(data_dir, 'X_trn.npy')
    # y_save_path = path.join(data_dir, 'Y_trn.npy')

    np.save(x_save_path, X)
    # np.save(y_save_path, Y)

    print(f'Final dataset sizes:\n  X: {X.shape}')
    # print(f"Final dataset sizes:\n  X: {X.shape}\n  Y: {Y.shape}")



if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', '-d', default='./data/processed/')
    parser.add_argument('--filename1', '-f1', required=True)
    parser.add_argument('--filename2', '-f2', required=True)
    parser.add_argument('--filename3', '-f3', required=True)

    params = parser.parse_args()

    concat_data(params.data_dir, params.filename1, params.filename2, params.filename3)

