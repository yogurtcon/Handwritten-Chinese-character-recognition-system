import os
import cv2
import numpy as np
import matplotlib.image as mi


dataset = []  # 数据集列表
labels = []  # 标签列表
label = 0  # 第一个标签


def load_data(filepath):
    # 遍历filepath下所有文件，包括子目录
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi+'/')
        if os.path.isdir(fi_d):
            global label
            load_data(fi_d)
            label += 1
        else:
            labels.append(label)
            img = mi.imread(fi_d[:-1])
            img2 = cv2.resize(img, (64, 64))  # (64,64,3)
            dataset.append(img2)

    return np.array(dataset), np.array(labels)

