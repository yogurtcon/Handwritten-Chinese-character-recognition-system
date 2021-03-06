# Introduction

针对低端计算机一般场景下的手写汉字识别，采用HWDB1.1数据集里面随机的100个汉字，使用机器学习框架Tensorflow，设计了基于卷积神经网络结构的手写汉字识别模型。实验结果表明，模型能够以很少的训练次数快速收敛，准确率达到93%以上。

Aiming at handwritten Chinese character recognition in low-end computer general scene, a handwritten Chinese character recognition model based on convolution neural network structure is designed by using 100 random Chinese characters in hwdb1.1 data set and tensorflow, a machine learning framework. The experimental results show that the model can converge quickly with few training times, and the accuracy is over 93%.

# Version

### v2.0 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v2.0

运行程序前要先创建original pic和cut pic两个文件夹，将需要识别的图片放入original pic文件夹，运行predict_v4.py文件会输出识别出的字，cut pic文件夹中会出现原图片切割出来的每个文字单独的图片

### v1.4 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v1.4

Chinese_recognition_model_v1.h5    epochs = 18

train loss: 0.06680139807916305     train accuracy: 0.97753626

test loss: 0.5099224895140645     test accuracy: 0.8906747102737427

Chinese_recognition_model_v2.h5    epochs = 18

loss: 0.2319 - accuracy: 0.9307 - val_loss: 0.2508 - val_accuracy: 0.9165

### v1.3 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v1.3

Chinese_recognition_model.h5    epochs:10

train loss: 0.11947370274117632     train accuracy: 0.9578175

test loss: 0.5367488763688817     test accuracy: 0.8849824070930481

Chinese_recognition_model_v2    epochs:20

loss: 0.2182 - accuracy: 0.9347 - val_loss: 0.4227 - val_accuracy: 0.9037

Chinese_recognition_model_v3    epochs:20

loss: 0.1170 - accuracy: 0.9640 - val_loss: 0.2514 - val_accuracy: 0.9299

### v1.2 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v1.2

train loss: 0.04     train accuracy: 0.99

val loss: 0.18     val accuracy: 0.96

test loss: 0.21     test accuracy: 0.95

### v1.1 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v1.1

test loss: 0.18     test accuracy: 0.96

### v1.0 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v1.0

test loss: 0.24 test accuracy: 0.94

### v0.9 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v0.9

test loss: 0.65 test accuracy: 0.82

# Tips

灰度化，归一化，数组降维等等操作都会导致特征丢失

Graying, normalization, array dimensionality reduction and other operations will lead to feature loss

并不是每一次训练都要做这些处理，每做一次处理都可能影响准确率

Not every training needs to do these processing, every processing may affect the accuracy rate

什么时候，需不需要做这些处理，需要视情况而定

When to do these things, do or not to do, it depends on the situation

# 感谢分享精神

参考 知乎用户：想飞的石头 的一篇文章 https://zhuanlan.zhihu.com/p/24698483

Reference to Zhihu user: 想飞的石头

作者处理好的数据，放到了云盘 https://pan.baidu.com/s/1o84jIrg#list/path=%2F ，char_dict是汉字和对应的数字label的记录

The data processed by the author is put into the cloud disk, and char_dict is the record of Chinese characters and corresponding digital labels

