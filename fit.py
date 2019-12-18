import get_model
import get_pyplot
import get_array_1
import get_array_2


epochs = 8  # 选择批次

model = get_model.get_model()  # 选择模型

# 加载训练数据和测试数据
(train_image, val_image, train_label, val_label) = get_array_1.load_data('data/train/')
(test_image, test_label) = get_array_2.load_data('data/test/')

# 训练, fit方法自带shuffle随机读取
history = model.fit(
    train_image, train_label, epochs=epochs, validation_data=(val_image, val_label))

# 测试, 单用evaluate方法不会自动输出数值，需要手动输出他返回的两个数值
test_scores = model.evaluate(test_image, test_label)

epochs_range = range(1, epochs+1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
test_loss = test_scores[0]
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
test_accuracy = test_scores[1]

# 将模型保存为 HDF5 文件
model.save('Chinese_recognition_model.h5')
print("save model: Chinese_recognition_model.h5")

# 绘制图表
get_pyplot.show(epochs_range, train_loss, val_loss, train_accuracy, val_accuracy)

#  打印得分
print('')
print('train loss:', train_loss[-1], '   ', 'train accuracy:', train_accuracy[-1])
print('val loss:', val_loss[-1], '   ', 'val accuracy:', val_accuracy[-1])
print('test loss:', test_loss, '   ', 'test accuracy:', test_accuracy)
print('')
