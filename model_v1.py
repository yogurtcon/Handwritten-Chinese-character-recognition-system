import keras.backend as k
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


def get_model():
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

    return model
