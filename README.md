# Version

### v1.3 ###

https://github.com/yogurtcon/Handwritten-Chinese-character-recognition-system/releases/tag/v1.3

Chinese_recognition_model.h5    epochs:10

train loss: 0.11947370274117632     train accuracy: 0.9578175
val loss: 0.47219873436071996     val accuracy: 0.8951436877250671
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

