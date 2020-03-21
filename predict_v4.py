import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

base_dir = "./original pic/"  # 将待识别的图片放入此文件夹
dst_dir = "./cut pic/"        # 切割过的图片生成于此文件夹
min_val = 10
min_range = 30

count = 0


def extract_peek(array_vals, minimun_val, minimun_range):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            if i - start_i >= minimun_range:
                end_i = i
                # print(end_i - start_i)
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


def cutImage(img, peek_range):
    global count
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            count += 1
            img1 = img[y:peek_range[1], x:vertical_range[1]]
            new_shape = (64, 64)
            img1 = cv2.resize(img1, new_shape)
            cv2.imwrite(dst_dir + str(count).zfill(5) + ".png", img1)  # zfill(x) 字符串中未满x个字符的话前面补零
            # cv2.rectangle(img, pt1, pt2, color)


for fileName in os.listdir(base_dir):
    img = cv2.imread(base_dir + fileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                               cv2.THRESH_BINARY_INV, 11, 2)
    horizontal_sum = np.sum(adaptive_threshold, axis=1)
    peek_ranges = extract_peek(horizontal_sum, min_val, min_range)
    line_seg_adaptive_threshold = np.copy(adaptive_threshold)
    for i, peek_range in enumerate(peek_ranges):
        x = 0
        y = peek_range[0]
        w = line_seg_adaptive_threshold.shape[1]
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek(
            vertical_sum, min_val, min_range)
        vertical_peek_ranges2d.append(vertical_peek_ranges)
    cutImage(img, peek_range)

model = tf.keras.models.load_model('Chinese_recognition_model_v2.h5')

files = os.listdir(dst_dir)
for fi in files:
    fi_d = os.path.join(dst_dir, fi + '/')

    img = mpimg.imread(fi_d[:-1])
    img2 = cv2.resize(img, (64, 64))
    img3 = np.zeros((1, img2.shape[0], img2.shape[1], img2.shape[2]))  # (1, 64, 64, 3)
    img3[0, :] = img2

    pre = model.predict(img3)  # 预测

    predicted_label = np.argmax(pre[0])
    class_names = ['一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '专', '且', '世', '丘', '丙', '业', '丛', '东', '丝',
                   '丢', '两', '严', '丧', '个', '丫', '中', '丰', '串', '临', '丸', '丹', '为', '主', '丽', '举', '乃', '久', '么', '义',
                   '之', '乌', '乍', '乎', '乏', '乐', '乒', '乓', '乔', '乖', '乘', '乙', '九', '乞', '也', '习', '乡', '书', '买', '乱',
                   '乳', '乾', '了', '予', '争', '事', '二', '于', '亏', '云', '互', '五', '井', '亚', '些', '亡', '亢', '交', '亥', '亦',
                   '产', '亨', '亩', '享', '京', '亭', '亮', '亲', '人', '亿', '什', '仁', '仅', '仆', '仇', '今', '介', '仍', '从', '仑',
                   '仓']

    print(class_names[predicted_label], end=' ')
    # print(100 * np.max(pre[0]))
