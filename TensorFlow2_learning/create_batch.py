import os

import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(0)

print('----------------- Reorganize the folder structure -------------------')

BASE_DIR = '/Users/tingxuanhu/Desktop/TensorFlow2_learning/data/'
names = ["Benign", "Normal", "Malignant"]

if not os.path.isdir(BASE_DIR + 'train/'):
    for name in names:
        os.makedirs(BASE_DIR + 'train/' + name)
        os.makedirs(BASE_DIR + 'val/' + name)
        os.makedirs(BASE_DIR + 'test/' + name)
#
# orig_folders = ['class1/', 'class2/', 'class3/']
# for folder_idx, folder in enumerate(orig_folders):
#     files = os.listdir(BASE_DIR + folder)
#     number_of_images = len([name for name in files])
#     n_train = int((number_of_images * 0.6) + 0.5)
#     n_valid = int((number_of_images * 0.25) + 0.5)
#     n_test = number_of_images - n_train - n_valid
#     print(number_of_images, n_train, n_valid, n_test)
#     for idx, file in enumerate(files):
#         file_name = BASE_DIR + folder + file
#         if idx < n_train:
#             shutil.copy(file_name, BASE_DIR + "train/" + names[folder_idx])
#         elif idx < n_train + n_valid:
#             shutil.copy(file_name, BASE_DIR + "val/" + names[folder_idx])
#         else:
#             shutil.copy(file_name, BASE_DIR + "test/" + names[folder_idx])

print('----------------- Generate batches with real-time data augmentation. -------------------')
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#    rotation_range=20,
#    horizontal_flip=True,
#    width_shift_range=0.2, height_shift_range=0.2,
#    shear_range=0.2, zoom_range=0.2

train_batches = train_gen.flow_from_directory(
    directory='/Users/tingxuanhu/Desktop/TensorFlow2_learning/data/train',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=True,
    color_mode='rgb',
    classes=names
)

val_batches = valid_gen.flow_from_directory(
    directory='/Users/tingxuanhu/Desktop/TensorFlow2_learning/data/val',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode='rgb',
    classes=names
)

test_batches = test_gen.flow_from_directory(
    directory='/Users/tingxuanhu/Desktop/TensorFlow2_learning/data/test',
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode='rgb',
    classes=names
)

# train_batch = train_batches[0]
# print('train_batch[0].shape-->', train_batch[0].shape)
# print('train_batch[1]-->', train_batch[1])
# test_batch = test_batches[0]
# print('test_batch[0].shape-->', test_batch[0].shape)
# print('test_batch[1]-->', test_batch[1])
#
#
# def show(batch, pred_labels=None):
#     matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#     plt.figure(figsize=(10, 10))
#     for i in range(4):
#         plt.subplot(2, 2, i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(batch[0][i], cmap=plt.cm.binary)
#         # The CIFAR labels happen to be arrays,
#         # which is why you need the extra index
#         lbl = names[int(batch[1][i])]
#         if pred_labels is not None:
#             lbl += "/ Pred:" + names[int(pred_labels[i])]
#         plt.xlabel(lbl)
#     plt.show()
#
#
# # show(test_batch)
# show(train_batch)


x = tf.keras.Input(shape=(256, 256, 3))
layer = tf.keras.layers.Conv2D(32, 3, 1, padding='valid', activation='relu')(x)
layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
layer = tf.keras.layers.Conv2D(64, 3, activation='relu')(layer)
layer = tf.keras.layers.MaxPool2D((2, 2))(layer)
layer = tf.keras.layers.Flatten()(layer)
layer = tf.keras.layers.Dense(64, activation='relu')(layer)
layer = tf.keras.layers.Dense(5)(layer)

model = tf.keras.Model(inputs=x, outputs=layer, name='conv_model')

print(model.summary())


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate=.001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# train
epochs = 30

# callbacks

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=2
)


history = model.fit(train_batches, validation_data=val_batches, epochs=epochs, callbacks=[early_stopping], verbose=1)

model.save('conv_model.h5')

# -------------- plot loss and acc -------------------
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend(fontsize=15)


model.evaluate(test_batches, verbose=2)










