#!/usr/bin/env python
# coding:utf8
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, plot_model
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tensorflow as tf
import seaborn as sns
import logging
import datetime
import sys

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)


# create train model
def modelcreate():
    model = Sequential([
        Conv1D(128, kernel_size, padding='same', input_shape=(data.shape[1], data.shape[2])),
        LeakyReLU(alpha=alpha),
        MaxPooling1D(4, padding='same'),
        Dropout(dropout),

        Conv1D(64, kernel_size, padding='same'),
        LeakyReLU(alpha=alpha),
        MaxPooling1D(4, padding='same'),
        Dropout(dropout),
        # Conv1D(32, kernel_size, padding='same'),
        # LeakyReLU(alpha=alpha),
        # MaxPooling1D(2, padding='same'),
        # Dropout(dropout),
        Flatten(),
        Dense(128),
        LeakyReLU(alpha=alpha),
        Dropout(dropout),
        Dense(64),
        LeakyReLU(alpha=alpha),
        Dropout(dropout),
        Dense(units=labels_train.shape[1], activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])
    return model


# model fit and save
def modelfit():
    model = modelcreate()
    # tensorflow实时日志
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fitlogspath + strftime, histogram_freq=1)

    # 定义回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(bestmodelfile, monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    starttime = BaseUtils.get_timestamp()
    # 训练模型
    history = model.fit(features_train, labels_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                        validation_data=(features_test, labels_test),
                        callbacks=[callbacks.History(), tensorboard_callback, early_stopping, model_checkpoint,
                                   reduce_lr])
    model.load_weights(bestmodelfile)
    endtime = BaseUtils.get_timestamp()
    model.save(modelfilename, save_format='h5')

    # 从history对象中提取训练和验证的指标
    accuracy = history.history['val_accuracy'][-1]  # 取验证集上最后一个epoch的准确率
    loss = history.history['val_loss'][-1]  # 取验证集上最后一个epoch的丢失率
    precision = history.history['val_precision'][-1]  # 取验证集上最后一个epoch的精确率
    recall = history.history['val_recall'][-1]  # 取验证集上最后一个epoch的召回率
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  # 计算F1分数
    # 输出指标
    print(f'最后一次迭代准确率: {accuracy:.4f}')
    print(f'最后一次迭代丢失率: {loss:.4f}')
    print(f'最后一次迭代精确率: {precision:.4f}')
    print(f'最后一次迭代召回率: {recall:.4f}')
    print(f'F1 分数: {f1:.4f}')
    loss, accuracy, precision, recall = model.evaluate(features_test, labels_test)
    # 20%模型测试
    print('20%%准确率测试: %f' % (accuracy * 100))
    print('20%%丢失率测试: %f' % (loss * 100))
    print('20%%精确度测试: %f' % (precision * 100))
    print('20%%召回率测试: %f' % (recall * 100))
    print(f'训练总时间:{str(endtime - starttime)}')

    plt.rcParams.update({'font.size': 15})
    # 绘制训练和验证的准确度以及损失
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('../sign-language-translate/test/sign-language.png')
    plt.show()

    # 预测
    y_pred = model.predict(features_test)
    y_pred_classes = y_pred.argmax(axis=1)
    y_true = labels_test.argmax(axis=1)
    # 获取标签名称
    target_names = encoder.classes_
    # 分类报告
    report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred_classes, target_names=target_names))
    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('../sign-language-translate/test/confusion_matrix.png')
    plt.show()
    # 绘制模型结构图
    plot_model(model, to_file='../sign-language-translate/test/model_structure.png', show_shapes=True,
               show_layer_names=True, dpi=300)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # tf.config.set_visible_devices(tf.config.list_physical_devices("GPU"))
    # print(tf.config.list_logical_devices())
    data = []
    labels = []
    strftime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # fit
    epochs = int(sys.argv[1])
    # batchsize
    batch_size = 32768
    # LeakyReLU
    alpha = 0.2
    # dropout
    dropout = 0.1
    # kernel_size
    kernel_size = 3
    bestmodelfile = './best_model.keras'
    modelfilename = '../sign-language-translate/test/sign-language.h5'
    pltpng = '../sign-language-translate/test/sign-language.png'
    npyfilespath = './npyfiles/'
    fitlogspath = './logs/fit/'
    # data
    # 初始化字典来存储数据
    label_data_dict = {}
    for filename in os.listdir(npyfilespath):
        if filename.endswith('.npy'):  # 确保只处理.npy文件
            # 提取不包含扩展名的文件名作为键
            word = filename[:-4]
            # 加载.npy文件并将数据存储到字典中
            label_data_dict[word] = np.load(os.path.join(npyfilespath, filename), allow_pickle=True)
    # 创建标签映射字典
    label_map = {i: label for i, label in enumerate(label_data_dict)}
    print(label_map)
    # 保存标签映射到文件
    with open('../sign-language-translate/test/sign-language-model.pkl', 'wb') as f:
        pickle.dump(label_map, f)

    for label, arr in label_data_dict.items():
        data.extend(arr)
        labels.extend([label] * len(arr))
        print(f"Label '{label}' corresponds to {arr.shape[0]} rows.")
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)
    # 使用Counter来计算每个标签的出现次数
    label_counts = Counter(labels)
    # 打印每个标签及其数量
    for label, count in label_counts.items():
        print(f"Label '{label}' appears {count} samples.")
    # 可选：打印总的标签数量
    print("Total number of samples:", len(labels))
    print("Total number of label:", len(label_counts))

    # 标签编码
    encoder = LabelEncoder()

    encoded_labels = to_categorical(encoder.fit_transform(labels))

    features_train, features_test, labels_train, labels_test = train_test_split(data, encoded_labels, test_size=0.2,
                                                                                random_state=42)
    # modelcreate()
    modelfit()
