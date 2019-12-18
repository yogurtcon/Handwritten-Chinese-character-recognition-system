import tensorflow as tf
import get_array_2
import numpy as np
import matplotlib.pyplot as plt


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

    print('预测结果:', class_names[predicted_label])


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(20), class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(20), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


class_names = ['líng', 'yī', 'èr', 'sān', 'sì', 'wǔ', 'liù', 'qī', 'bā', 'jiǔ', 'zhào', 'qìng', 'xué', 'yuàn',
               'jì', 'suàn', 'jī', 'yáng', 'xiān', 'shēng']

model = tf.keras.models.load_model('Chinese_recognition_model.h5')

(test_image, test_label) = get_array_2.load_data('data/test/')

predictions = model.predict(test_image)

i = 1000
plt.subplot(2, 1, 1)
plot_image(i, predictions[i], test_label, test_image)
plt.subplot(2, 1, 2)
plot_value_array(i, predictions[i],  test_label)
plt.show()
