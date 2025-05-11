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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import SGD, Adam
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

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
N_ENC = 64
N_HIDDEN = 256

N_CONTEXT = 60 + 1  # The number of frames in the context

def rmse_loss(y_true, y_pred):
  err = y_true - y_pred
  loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(err)))
  return loss


def train(model_file):
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

    print(X.shape)
    print(Y.shape)

    # Split on training and validation
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Define Keras model

    model = Sequential()
    model.add(TimeDistributed(Dense(N_ENC), input_shape=(N_CONTEXT, N_INPUT)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(TimeDistributed(Dense(N_HIDDEN)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(GRU(N_HIDDEN, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(N_OUTPUT))
    model.add(Activation('linear'))

    print(model.summary())

    # optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
    # model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss=rmse_loss, optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['accuracy'])
    # model.compile(loss=tf.keras.losses.CosineSimilarity(), optimizer=optimizer, metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_validation, Y_validation))

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
    model_file = sys.argv[1]
    
    if not model_file.endswith(".hdf5"):
        model_file += ".hdf5"

    train(model_file)