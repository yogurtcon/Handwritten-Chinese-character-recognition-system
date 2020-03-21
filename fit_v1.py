# 直接一次性加载全部数据

import get_model
import get_pyplot
import get_train_array
import get_test_array

epochs = 20  # 选择批次

model = get_model.get_model()  # 选择模型

# 加载训练数据和测试数据
(train_image, val_image, train_label, val_label) = get_train_array.load_data('data/train/')
(test_image, test_label) = get_test_array.load_data('data/test/')

# 训练, fit方法自带shuffle随机读取
history = model.fit(
    train_image, train_label, epochs=epochs, validation_data=(val_image, val_label))

# 测试, 单用evaluate方法不会自动输出数值，需要手动输出他返回的两个数值
test_scores = model.evaluate(test_image, test_label)

epochs_range = range(1, epochs+1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
test_loss = test_scores[0]
train_acc = history.history['acc']
val_acc = history.history['val_acc']
test_acc = test_scores[1]

# 将模型保存为 HDF5 文件
model.save('Chinese_recognition_model_v1.h5')
print("save model: Chinese_recognition_model_v1.h5")

# 绘制图表
get_pyplot.show(epochs_range, train_loss, val_loss, train_acc, val_acc, 'Model_score_v1')

#  打印得分
print('')
print('train loss:', train_loss[-1], '   ', 'train accuracy:', train_acc[-1])
print('val loss:', val_loss[-1], '   ', 'val accuracy:', val_acc[-1])
print('test loss:', test_loss, '   ', 'test accuracy:', test_acc)
print('')
