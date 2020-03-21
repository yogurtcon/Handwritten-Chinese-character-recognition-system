import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

model = tf.keras.models.load_model('Chinese_recognition_model_v2.h5')

img = mpimg.imread('data/test/00034/2190.png')
img2 = cv2.resize(img, (64, 64))
img3 = np.zeros((1, img2.shape[0], img2.shape[1], img2.shape[2]))  # (1, 64, 64, 3)
img3[0, :] = img2

pre = model.predict(img3)  # 预测

predicted_label = np.argmax(pre[0])

class_names = ['一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '专', '且', '世', '丘', '丙', '业', '丛', '东', '丝', '丢',
               '两', '严', '丧', '个', '丫', '中', '丰', '串', '临', '丸', '丹', '为', '主', '丽', '举', '乃', '久', '么', '义', '之', '乌',
               '乍', '乎', '乏', '乐', '乒', '乓', '乔', '乖', '乘', '乙', '九', '乞', '也', '习', '乡', '书', '买', '乱', '乳', '乾', '了',
               '予', '争', '事', '二', '于', '亏', '云', '互', '五', '井', '亚', '些', '亡', '亢', '交', '亥', '亦', '产', '亨', '亩', '享',
               '京', '亭', '亮', '亲', '人', '亿', '什', '仁', '仅', '仆', '仇', '今', '介', '仍', '从', '仑', '仓']

print(class_names[predicted_label])
print(100 * np.max(pre[0]))
