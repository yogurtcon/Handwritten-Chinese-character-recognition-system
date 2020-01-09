import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg


model = tf.keras.models.load_model('Chinese_recognition_model.h5_v1')

img = mpimg.imread('data/test/00011/522.png')
img2 = cv2.resize(img, (64, 64))
img3 = np.zeros((1, img2.shape[0], img2.shape[1], img2.shape[2]))
img3[0, :] = img2

pre = model.predict(img3)  # 预测

predicted_label = np.argmax(pre[0])

class_names = list(range(100))

print(class_names[predicted_label])
print(100 * np.max(pre[0]))

