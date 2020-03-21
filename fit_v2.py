# 分批次加载数据，无灰度化处理，图片格式为(64, 64, 3)

from __future__ import print_function
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model, load_model
import get_pyplot
import get_model

data_dir = 'data'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

# dimensions of our images.
img_width, img_height = 64, 64  # 宽高64
charset_size = 100  # 100分类
nb_nb_epoch = 10  # 训练30次


def train(model):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=50,
        # color_mode="grayscale",
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=33,
        # color_mode="grayscale",
        class_mode='categorical')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=476,
                                  nb_epoch=nb_nb_epoch,
                                  validation_data=validation_generator,
                                  validation_steps=181,)

    epochs_range = range(1, nb_nb_epoch + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    # 绘制图表
    get_pyplot.show(epochs_range, train_loss, val_loss, train_acc, val_acc, 'Model_score_v2')


def build_model(include_top=True, input_shape=(64, 64, 3), classes=charset_size):
    img_input = Input(shape=input_shape)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dropout(0.05)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='model')
    model.summary()
    return model


model = build_model()
# model = load_model("./model.h5")


# model = get_model.get_model()


train(model)
model.save("Chinese_recognition_model_v2.h5")
