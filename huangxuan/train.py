#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:37:28 2023

@author: randong
"""
import itertools
import numpy as np
import torch
import os
from pathlib import Path
from torch import nn
from torch.utils.data.dataloader import DataLoader
import time
import model
import matplotlib.pylab as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")
param_dir = Path("./params")
param_dir.mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available():
    print("CUDA is available.")


# 1DCNN
# 根据subject获取标签的方法
def get_subject_dirs(directory):
    subject_dirs = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            if dirname.startswith("subject"):
                subject_dirs.append(os.path.join(dirpath, dirname))
    subject_dirs.sort()  # 按文件名排序
    return subject_dirs


def get_files(directory):
    # 选择摄像头
    # cam3_directory = os.path.join(directory, 'cam4')
    # cam4_directory = os.path.join(directory, 'cam4')

    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.npy'):
                files.append(os.path.join(dirpath, filename))

    # for dirpath, dirnames, filenames in os.walk(cam3_directory):#directory):去掉选择摄像头和(camX_directory)定义，则获取全部NPY文件
    #     for filename in filenames:
    #         if filename.endswith('.npy'):
    #             files.append(os.path.join(dirpath, filename))
    files.sort()
    return files


datapath = r"C:\Work HX\博士相关\博士论文投稿1\代码，数据及处理\Code\1Dcnn OD\Original data LA cam5"
subject_dirs = get_subject_dirs(datapath)

win = 30  # 后续
dof = 54  # 重点

data = np.empty([0, dof, win])
label = np.empty([0])

for i, subject_dir in enumerate(subject_dirs):
    npy_files = get_files(subject_dir)
    for npy_file in npy_files:
        tmp = np.load(npy_file, allow_pickle=True)
        data = np.concatenate([data, tmp], axis=0)
        label = np.concatenate([label, np.array([i] * len(tmp))], axis=0)
        print(data.shape)

# 接下来是董老师的代码
# normalization
norm_path = datapath + "/norm.npz"

dataNorm = np.concatenate(data, axis=-1)
print("data shape 1/3:", dataNorm.shape)
# [V, sumT, J * 2] / [sumT, J * 3/4 + 4]
dataNorm = dataNorm.swapaxes(-1, -2)
print("data shape 2/3:", dataNorm.shape)
# [V * sumT, J * 2] / [sumT, J * 3/4 + 4]
dataNorm = dataNorm.reshape((-1, dataNorm.shape[-1]))
print("data shape 3/3:", dataNorm.shape)

mean = np.mean(dataNorm, axis=0)
std = np.std(dataNorm, axis=0, dtype=np.float32)  # 修改3，数据类型设置为float32
std[np.where(std == 0)] = 1e-9
np.savez(norm_path, mean=mean, std=std)
print("mean and std saved at {}".format(norm_path))

dataN = (data - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]

# data to device
dataN = torch.from_numpy(dataN.astype(np.float32)).clone()
label = torch.from_numpy(label.astype(np.int64)).clone()

dataset = torch.utils.data.TensorDataset(dataN, label)  # 重要

# acceleration
torch.backends.cudnn.benchmark = True

# test train data

ratio = 0.8
batch_size = 1024  # 重点

eval_interval = 50

train_dataset = dataset[:int(len(dataset) * ratio)][0]
valid_dataset = dataset[int(len(dataset) * ratio):][0]

train_dataset, valid_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset) * ratio), len(dataset) - int(len(dataset) * ratio)], torch.Generator().manual_seed(0))

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True,
    shuffle=True)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, drop_last=True,
    shuffle=True)


# initialize weights
def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.zeros_(m.bias.data)


encoder = model.AutoEncoder(dof).to(device)  # (63)
encoder.apply(weights_init)

optim = torch.optim.Adam(encoder.parameters(), lr=0.00001)

loss_f = nn.CrossEntropyLoss()
iter_ = 0
num = len(train_dataset)

if __name__ == "__main__":
    lossTrain = []
    lossTest = []
    accTrain = []
    accTest = []
    epoch_time = []  # added
    total_start_time = time.time()  # added

    iter_ = 0
    num_epochs = 1000  # 重点

    for epoch in range(num_epochs):
        train_iters = 0
        train_err_ = 0
        correct = 0
        epoch_start_time = time.time()

        for data in train_dataloader:
            data, label = data
            data = data.to(device)
            label = label.to(device)

            optim.zero_grad()

            train_out = encoder(data)
            train_err = loss_f(train_out, label)
            train_err_ += train_err.item()

            train_err.backward()
            optim.step()

            iter_ += 1
            train_iters += 1

            pred = train_out.argmax(1)
            correct += pred.eq(label.view_as(pred)).sum().item()

        train_err_ /= train_iters
        epoch_end_time = time.time()
        epoch_time.append(epoch_end_time - epoch_start_time)
        print(f"Epoch: {epoch} Train Loss: {train_err_:.2f} Time: {epoch_end_time - epoch_start_time:.2f}s")

        accTrain.append(100 * correct / (len(train_dataloader) * batch_size))

        valid_err_ = 0
        valid_iters = 0
        correct = 0

        for valid_data in valid_dataloader:
            valid_data, valid_label = valid_data

            valid_data = valid_data.to(device, non_blocking=True)
            valid_label = valid_label.to(device, non_blocking=True)

            with torch.no_grad():
                valid_out = encoder(valid_data)
                valid_err = loss_f(valid_out, valid_label)

            valid_iters += 1
            valid_err_ += valid_err.item()

            pred = valid_out.argmax(1)
            correct += pred.eq(label.view_as(pred)).sum().item()

        valid_err_ /= valid_iters

        print(f"Epoch: {epoch} Validation Loss: {valid_err_:.2f}")
        print(
            f"Accuracy: {100 * correct / (len(valid_dataloader) * batch_size)}% ({correct}/{(len(valid_dataloader) * batch_size)})")

        accTest.append(100 * correct / (len(valid_dataloader) * batch_size))

        lossTrain.append(train_err_)
        lossTest.append(valid_err_)

        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
                        'iters': iter_,
                        'time': time.time(),
                        'train_loss': train_err_,
                        'valid_loss': valid_err_,
                        'network': encoder.state_dict(),
                        'optim': optim.state_dict()},
                       os.path.join(param_dir, f'{iter_:06d}.pth'))

        if epoch == num_epochs - 1:
            break

    total_end_time = time.time()
    print(f"Total time: {total_end_time - total_start_time:.2f}s")

fig_path = str(param_dir)

# 绘制时间曲线图
plt.plot(range(num_epochs), epoch_time, '-')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Time per Epoch')
plt.savefig(fig_path + "/Time.pdf")
plt.show()

# save timedata
np.save(fig_path + "/time", np.array([epoch_time]))

# plot
plt.plot(lossTrain, label="Training", lw=1)
plt.plot(lossTest, label="Testing", lw=1)
# plt.loglog()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0, 3)
plt.savefig(fig_path + "/LOSS.pdf")
plt.show()

# save lossdata
np.save(fig_path + "/loss", np.array([lossTrain, lossTest]))

# plot
plt.plot(accTrain, label="Training", lw=1)
plt.plot(accTest, label="Testing", lw=1)
# plt.loglog()
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend()
plt.savefig(fig_path + "/ACC.pdf")
plt.show()

# save lossdata
np.save(fig_path + "/acc", np.array([accTrain, accTest]))
print("训练完成！")
