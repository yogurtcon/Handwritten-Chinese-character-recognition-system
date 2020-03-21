import random

import tensorflow as tf
import get_test_array
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 忽略警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


class_names = list(range(100))

model = tf.keras.models.load_model('Chinese_recognition_model_v2.h5')

(test_image, test_label) = get_test_array.load_data('data/test/')

index = [x for x in range(len(test_label))]
random.shuffle(index)
test_image = test_image[index]
test_label = test_label[index]

predictions = model.predict(test_image)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    plot_image(i, predictions[i], test_label, test_image)
plt.tight_layout()
plt.show()
