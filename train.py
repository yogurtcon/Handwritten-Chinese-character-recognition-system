import get_array_1
import get_array_2
import get_pyplot
import model_v1
import model_v2


epochs = 3                      # 选择批次
model = model_v2.get_model()    # 选择模型

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
test_acc = test_scores[1]

# 将模型保存为 HDF5 文件
model.save('Chinese_recognition_model.h5')
print('')
print("save model: Chinese_recognition_model.h5")

# 绘制图表
get_pyplot.show(
    epochs_range, train_loss, val_loss, train_accuracy, val_accuracy)

get_pyecharts.line_smooth(
    epochs_range, train_loss, val_loss, train_accuracy, val_accuracy)

#  打印得分
print('')
print('train loss:', train_loss[-1], '   ', 'train acc:', train_accuracy[-1])
print('val loss:', val_loss[-1], '   ', 'val acc:', val_accuracy[-1])
print('test loss:', test_loss, '   ', 'test acc:', test_acc)
print('')
