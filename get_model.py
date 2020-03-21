import keras.backend as k
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D


def get_model():

    k.clear_session()

    # 创建一个新模型
    model = Sequential()

    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(64, 64, 3)))  # 64 64 3
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(100, activation='softmax'))

    model.summary()

    # 选择优化器和损失函数
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
