import numpy as np
import tensorflow as tf
import get_array_1
import get_array_2
import get_pyplot
import matplotlib.pyplot as plt


# 加载训练数据和测试数据
(train_image, val_image, train_label, val_label) = get_array_1.load_data('data/train/')
(test_image, test_label) = get_array_2.load_data('data/test/')


train_image_2, test_image_2, val_image_2 = \
    train_image / 255.0, test_image / 255.0, val_image/255.0


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


train_image_3 = rgb2gray(train_image_2)
test_image_3 = rgb2gray(test_image_2)
val_image_3 = rgb2gray(val_image_2)


train_image_4 = train_image_3.reshape(1786, 4096)
test_image_4 = test_image_3.reshape(595, 4096)
val_image_4 = val_image_3.reshape(596, 4096)


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',       # 为训练选择优化器和损失函数
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 100

history = model.fit(
    train_image_4, train_label, epochs=epochs, validation_data=(val_image_4, val_label))  # 训练

model.evaluate(test_image_4, test_label)  # 测试

epochs_range = range(1, epochs+1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']
# 绘制图表
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.savefig('acc_loss_pyplot_1D.png')
plt.show()

# 将模型保存为 HDF5 文件
model.save('Chinese_recognition_model_1D.h5')

print("模型已保存")
