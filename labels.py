import get_train_array
import get_test_array

# 加载训练数据和测试数据
(train_image, val_image, train_label, val_label) = get_train_array.load_data('data/train/')
(test_image, test_label) = get_test_array.load_data('data/test/')

print(train_label[0: 100])
print('')
print(val_label[0: 100])
print('')
print(test_label[0: 100])
