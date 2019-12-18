from __future__ import print_function
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model, load_model

data_dir = 'data'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')

# dimensions of our images.
img_width, img_height = 64, 64
charset_size = 20
nb_nb_epoch = 20


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
        batch_size=1024,
        color_mode="grayscale",
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=1024,
        color_mode="grayscale",
        class_mode='categorical')

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit_generator(train_generator,
                        nb_epoch=nb_nb_epoch,
                        validation_data=validation_generator,)


def build_model(include_top=True, input_shape=(64, 64, 1), classes=charset_size):
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
    return model


model = build_model()
# model = load_model("./model.h5")
train(model)
model.save("Chinese_recognition_model_v2.h5")
