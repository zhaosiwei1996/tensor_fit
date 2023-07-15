import numpy as np
import os
import pandas as pd


# 预处理mediapipe数据

def get_xlsx_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.xlsx')]


def pad_to_window(slice, window):
    def get_reflection(src, tlen):  # return [src-reversed][src][src-r]...
        x = src.copy()
        x = np.flip(x, axis=0)
        ret = x.copy()
        while len(ret) < tlen:
            x = np.flip(x, axis=0)
            ret = np.concatenate((ret, x), axis=0)
        ret = ret[:tlen]
        return ret

    if len(slice) >= window:
        return slice
    left_len = (window - len(slice)) // 2 + (window - len(slice)) % 2
    right_len = (window - len(slice)) // 2
    left = np.flip(get_reflection(np.flip(slice, axis=0), left_len), axis=0)
    right = get_reflection(slice, right_len)
    slice = np.concatenate([left, slice, right], axis=0)
    assert len(slice) == window
    return slice


def divide_clip(input, window, window_step, divide):
    if not divide:  # return the whole clip
        t = ((input.shape[0]) // 4) * 4 + 4
        t = max(t, 12)
        if input.shape[0] < t:
            input = pad_to_window(input, t)
        return [input]
    windows = []
    for j in range(0, len(input) - window + 1, window_step):
        slice = input[j: j + window].copy()  # remember to COPY!!
        if len(slice) < window:
            slice = pad_to_window(slice, window)
        windows.append(slice.T)
    return windows


if __name__ == '__main__':
    # 25码率
    win = 25
    # 步长 帧率 5
    step = 5

    datapath = r"./npyfile/coverfile/"

    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith('.xlsx'):
                item = os.path.join(root, file)
                data = pd.read_excel(item, header=None, engine="openpyxl").iloc[1:, 1:].values
                data = divide_clip(data, win, step, True)
                np.save(os.path.join(root, file[:-5]), data)
                print(f"{os.path.join(root, file[:-5])}.npy has been saved!")
