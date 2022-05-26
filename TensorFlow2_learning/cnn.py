import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib_inline.backend_inline

cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.shape)  # 50000, 32, 32, 3

# Normalize
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# def show():
#     # matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#     plt.figure(figsize=(10, 10))
#     for i in range(16):
#         plt.subplot(4, 4, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(train_images[i], cmap=plt.cm.binary)
#         # The CIFAR labels happen to be arrays,
#         # which is why you need the extra index
#         plt.xlabel(class_names[train_labels[i][0]])
#     plt.show()


# show()

# model...
model = tf.keras.models.Sequential()
# tf.keras.layers.Conv2D ->kernels  kernel_size
model.add(tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPool2D(2, 2))  # default
model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))  # set no Softmax here because the following loss from_logits = True

# loss and optimizer
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate=.001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs)

# evaluate
model.evaluate(test_images, test_labels, batch_size=batch_size)


model.save_weights("nn_weights.h5")

# if you want to load weights, please to initialize model first:
# model = keras.Sequential([...])
model.load_weights("nn_weights.h5")
































