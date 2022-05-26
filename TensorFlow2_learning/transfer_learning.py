import tensorflow as tf

# base_model = tf.keras.applications.VGG16()
#
# x = base_model.layers[-2].output
# new_outputs = tf.keras.layers.Dense(1)(x)
#
# new_model = tf.keras.Model(inputs=base_model.inputs, outputs=new_outputs)

vgg_model = tf.keras.applications.vgg16.VGG16()
print(type(vgg_model))
print(vgg_model.summary())

for layer in vgg_model.layers:
    layer.trainable = False
print(vgg_model.summary())

layer = tf.keras.layers.Dense(5)(vgg_model.output)
model = tf.keras.Model(inputs=vgg_model.input, outputs=layer)

print(model.summary())

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = tf.keras.optimizers.Adam(learning_rate=.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# get the preprocessing function of this model
pre_input = tf.keras.applications.vgg16.preprocess_input

# generate batches of tensor image data with real-time data augmentation
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=pre_input)
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=pre_input)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=pre_input)

names = ["Benign", "Normal", "Malignant"]

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


epochs = 30

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=2
)

model.fit(train_batches, validation_data=val_batches, callbacks=[early_stopping], epochs=epochs, verbose=2)


model.evaluate(test_batches, verbose=2)




















