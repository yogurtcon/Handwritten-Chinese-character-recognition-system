import keras.backend as k
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import get_array_1
import get_array_2

# 加载训练数据和测试数据
(train_image, train_label) = get_array_1.load_data('data/train/')
(test_image, test_label) = get_array_2.load_data('data/test/')


input_shape = (64, 64, 3)
k.clear_session()

# 创建一个新模型
model = Sequential()

model.add(Conv2D(64, [3, 3], activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D([2, 2], [2, 2], padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, [3, 3], activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D([2, 2], [2, 2], padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, [3, 3], activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D([2, 2], [2, 2], padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

# 选择优化器和损失函数
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练, fit方法自带shuffle随机读取数据集
model.fit(train_image, train_label, epochs=3)

# 测试, 单用evaluate方法不会自动输出数值，要手动将其返回的两个数值
test_scores = model.evaluate(test_image, test_label)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# 将模型保存为 HDF5 文件
model.save('Chinese_recognition_model.h5')

print("模型已保存")
