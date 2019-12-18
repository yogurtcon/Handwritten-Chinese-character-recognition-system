import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg


class_names = ['líng', 'yī', 'èr', 'sān', 'sì', 'wǔ', 'liù', 'qī', 'bā', 'jiǔ', 'zhào', 'qìng', 'xué', 'yuàn',
               'jì', 'suàn', 'jī', 'yáng', 'xiān', 'shēng']

model = tf.keras.models.load_model('Chinese_recognition_model.h5')

img = mpimg.imread('data/test/r/29.png')
img2 = cv2.resize(img, (64, 64))
img3 = np.zeros((1, img2.shape[0], img2.shape[1], img2.shape[2]))
img3[0, :] = img2

pre = model.predict_classes(img3)  # 预测
result = class_names[pre[0]]
print('预测结果:', result)
