#

import get_model
import get_train_array
import get_test_array
import time
from keras import models


model = get_model.get_model()  # 选择模型

# 加载训练数据和测试数据
(train_image, val_image, train_label, val_label) = get_train_array.load_data('data/train/')
(test_image, test_label) = get_test_array.load_data('data/test/')

# 训练, fit方法自带shuffle随机读取
model.fit(train_image, train_label, validation_data=(val_image, val_label))

test_scores = model.evaluate(test_image, test_label)
test_accuracy = test_scores[1]

# 将模型保存为 HDF5 文件
strTime = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
print('Time:', strTime, '   ', 'Test Accuracy:', test_accuracy)
model.save('Chinese_recognition_model_v3.h5')


for i in range(1, 8):
    print()
    model = models.load_model('Chinese_recognition_model_v3.h5')
    model.fit(train_image, train_label)
    test_scores = model.evaluate(test_image, test_label)
    test_accuracy = test_scores[1]
    strTime=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    print('Time:', strTime, '   ', 'Test Accuracy:', test_accuracy)
    model.save('Chinese_recognition_model_v3.h5')
    if test_accuracy > 0.95:
        break
