import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg


class_names = ['ling', 'yi', 'er', 'san', 'si', 'wu', 'liu', 'qi', 'Ba', 'jiu']

model = tf.keras.models.load_model('Chinese_recognition_model.h5')  # 加载模型

img1 = mpimg.imread('data/test/0/11007.png')  # 加载图片
img2 = cv2.resize(img1, (64, 64))  # 图片缩放

src = img2
img3 = cv2.cvtColor(src, cv2.COLOR_RGBA2BGR)  # 图片通道数定为三

img4 = np.zeros((1, img3.shape[0], img3.shape[1], img3.shape[2]))
img4[0, :] = img3

pre = model.predict_classes(img4)  # 预测
result = class_names[pre[0]]
print('预测结果:', result)
