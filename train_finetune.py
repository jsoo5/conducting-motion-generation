"""
This is the main script for the training.
It contains speech-motion neural network implemented in Keras
This script should be used to train the model, as described in READ.me
"""

import sys
from os.path import join

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Dense, Activation, Dropout, Input,
    SimpleRNN, LSTM, GRU,
    TimeDistributed, Bidirectional,
    BatchNormalization
    )
from tensorflow.keras.optimizers import SGD, Adam

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
# from matplotlib import plt

# Check if script get enough parameters
if len(sys.argv) < 6:
        raise ValueError(
           'Not enough paramters! \nUsage : python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE (DIM)')
ENCODED = sys.argv[5].lower() == 'true'

if ENCODED:
    if len(sys.argv) < 7:
        raise ValueError(
           'Not enough paramters! \nUsage : python train.py MODEL_NAME EPOCHS DATA_DIR N_INPUT ENCODE DIM')
    else:    
        N_OUTPUT = int(sys.argv[6])  # Representation dimensionality
else:
    N_OUTPUT = 498  # Number of Gesture Features


EPOCHS = int(sys.argv[2])
DATA_DIR = sys.argv[3]
N_INPUT = int(sys.argv[4])  # Number of input features

BATCH_SIZE = 1024
# BATCH_SIZE = 512
N_ENC = 64
N_HIDDEN = 256

N_CONTEXT = 60 + 1  # The number of frames in the context


def base_model():
    # Define Keras model

    model = Sequential()
    # inputs = Input(shape=(N_CONTEXT, N_INPUT))
    model.add(TimeDistributed(Dense(N_ENC), input_shape=(N_CONTEXT, N_INPUT)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.2))

    model.add(GRU(N_HIDDEN, return_sequences=False, reset_after=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.3))

    model.add(Dense(N_OUTPUT))
    model.add(Activation('linear'))

    return model


def train_fine_tuning(model_file):
    """
    Train a neural network to take speech as input and produce gesture as an output

    Args:
        model_file: file to store the model

    Returns:

    """

    # Get the data
    X = np.load(join(DATA_DIR, 'X_train.npy'))
    _X = np.load(join(DATA_DIR, 'X_dev.npy'))
    X = np.concatenate((X, _X), axis=0)


    if ENCODED:
        # If we learn speech-representation mapping we use encoded motion as output
        Y = np.load(join(DATA_DIR, str(N_OUTPUT), 'Y_train_encoded.npy'))
        _Y = np.load(join(DATA_DIR, str(N_OUTPUT), 'Y_dev_encoded.npy'))
        Y = np.concatenate((Y, _Y), axis=0)

        # Correct the sizes
        train_size = min(X.shape[0], Y.shape[0])
        X = X[:train_size]
        Y = Y[:train_size]

    else:
        Y = np.load(join(DATA_DIR, 'Y_train.npy'))
        _Y = np.load(join(DATA_DIR, 'Y_dev.npy'))
        Y = np.concatenate((Y, _Y), axis=0)

    print('X.shape: ', X.shape)
    print('Y.shape: ', Y.shape)

    # N_train = int(len(X)*0.8)
    # N_validation = len(X) - N_train

    # Split on training and validation
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1, random_state=42)

    pretrained_model = base_model()
    pretrained_model.load_weights('model_500_weights.hdf5')
    # pretrained_model.trainable = False


    model = Sequential()

    model.add(TimeDistributed(Dense(N_ENC), input_shape=(N_CONTEXT, N_INPUT)))

    for layer in pretrained_model.layers[1:-2]:
        layer.trainable = False
        model.add(layer)
    #
    # for layer in pretrained_model.layers[1:-2]:
    #     if isinstance(layer, BatchNormalization):
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True
    #     model.add(layer)
    #
    # for layer in pretrained_model.layers[-2:]:
    #     layer.trainable = True
    #     model.add(layer)
    #
    # for layer in pretrained_model.layers[12:-2]:
    #     layer.trainable = False
    #     model.add(layer)

    # for layer in pretrained_model.layers[1:-2]:
    #     if isinstance(layer, GRU):
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True
    #     model.add(layer)

    model.add(Dense(N_OUTPUT))
    model.add(Activation('linear'))

    model.layers[0].trainable = False
    # model.layers[1].trainable = False

    for layer in model.layers:
        print(layer.trainable)

    optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
    # optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=100, validation_data=(X_validation, Y_validation))


    # for layer in model.layers:
    #     if isinstance(layer, BatchNormalization):
    #         layer.trainable = True

    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer = Dropout(0.1)

        layer.trainable = True
        print(layer.trainable)


    # optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    # predicted = np.array(model.predict(input_file))
    # print("Predicted shape is: ", predicted.shape)


    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS-100, validation_data=(X_validation, Y_validation))

    model.save(model_file)

    # Save convergence results into an image
    plt.figure(figsize=(21, 9))

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['acc'], linewidth=3, label='train', color='red')
    plt.plot(hist.history['val_acc'], linewidth=3, label='valid', color='green')
    plt.grid()
    plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()
    plt.savefig(model_file.replace('hdf5', 'png'))


if __name__ == "__main__":
    # pretrained_model_file = sys.argv[7]
    model_file = sys.argv[1]

    if not model_file.endswith(".hdf5"):
        model_file += ".hdf5"

    train_fine_tuning(model_file)

