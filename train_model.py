#!/usr/bin/env python
# coding:utf8
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import *
import numpy as np
import tensorflow as tf
import logging
import datetime
import sys

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)
label_encoder = LabelEncoder()


def modelcreate():
    model = Sequential([
        Conv1D(256, kernel_size, padding='same', input_shape=(data.shape[1], data.shape[2])),
        LeakyReLU(alpha=0.3),
        MaxPooling1D(2, padding='same'),
        Dropout(dropout),

        Conv1D(128, kernel_size, padding='same'),
        LeakyReLU(alpha=0.3),
        MaxPooling1D(2, padding='same'),
        Dropout(dropout),

        Conv1D(64, kernel_size, padding='same'),
        LeakyReLU(alpha=0.3),
        MaxPooling1D(2, padding='same'),
        Dropout(dropout),

        Flatten(),
        Dropout(dropout),

        Dense(128),
        LeakyReLU(alpha=0.3),
        Dropout(dropout),

        Dense(64),
        LeakyReLU(alpha=0.3),
        Dropout(dropout),

        Dense(units=len(label_encoder.classes_), activation='softmax'),
        # Dropout(0.3),
    ])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def modelfit():
    model = modelcreate()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fitlogspath + strftime, histogram_freq=1)
    starttime = BaseUtils.get_timestamp()
    history = model.fit(features_train, encoded_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(features_test, label_encoder.fit_transform(labels_test)),
                        callbacks=[callbacks.History(), tensorboard_callback])

    model.save(modelpath)
    loss, accuracy = model.evaluate(features_test, label_encoder.fit_transform(labels_test))
    endtime = BaseUtils.get_timestamp()
    logging.info('Test Accuracy: %f' % (accuracy * 100))
    logging.info('Test loss: %f' % (loss * 100))
    logging.info(f'fix totaltime:{str(endtime - starttime)}')


if __name__ == '__main__':
    # tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
    # print(tf.config.list_logical_devices())
    data = []
    labels = []
    strftime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # fit
    epochs = int(sys.argv[1])
    # batchsize
    batch_size = 4096
    # dropout
    dropout = 0.5
    # kernel_size
    kernel_size = 11
    npyfilepath = '/Users/wubo/Downloads/zhaosiwei/npyfiles/'
    fitlogspath = '/Users/wubo/Downloads/zhaosiwei/logs/fit/'
    modelpath = '/Users/wubo/Downloads/zhaosiwei/sign-language-model.h5_{}'.format(strftime)
    # data
    label_data_dict = {
        'before': np.load(f'{npyfilepath}before.npy', allow_pickle=True),
        'book': np.load(f'{npyfilepath}book.npy', allow_pickle=True),
        'chair': np.load(f'{npyfilepath}chair.npy', allow_pickle=True),
        'computer': np.load(f'{npyfilepath}computer.npy', allow_pickle=True),
        'drink': np.load(f'{npyfilepath}drink.npy', allow_pickle=True),
        'go': np.load(f'{npyfilepath}go.npy', allow_pickle=True)
    }
    for label, arr in label_data_dict.items():
        data.extend(arr)
        labels.extend([label] * len(arr))
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)
    print(data.shape)
    features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2,
                                                                                random_state=42)
    encoded_labels = label_encoder.fit_transform(labels_train)
    modelfit()
