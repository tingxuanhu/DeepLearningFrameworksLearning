import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# normalize: 0, 255 -> 0, 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(x_train[i], cmap='gray')
# plt.show()

# -------------- model building option 1---------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
])

print(model.summary())  # see the model's structure


# -------------- model building option 2 ---------------
# another way to build the Sequential model:
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10))


#  -------------- loss and optimizer ---------------
# y = 0  --> y = [1, 0, 0, 0, 0, 0, 0, 0, 0 ....]   one-hot
# Whether to interpret y_pred as a tensor of logit values.
# By default, we assume that y_pred contains probabilities (i.e., values in [0, 1]).
# from_logits = False as default
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
metrics = ["accuracy"]
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


# ---------------- training -------------------
batch_size = 64
epochs = 2

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)
# verbose = 0 --> no record
# verbose = 1 --> progress bar
# verbose = 2 --> normal logging


# --------------- evaluation ------------------
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)


# --------------- predictions ----------------
# -- option 1 --
probability_model = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Softmax()   # because the model's loss from_logits=True, so here we need softmax
])
predictions = probability_model(x_test)
pred0 = predictions[0]
label0 = np.argmax(pred0)

print(predictions)
print(pred0)
print(label0)


# -- option 2 --
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)


# -- option 3   recommended --
predictions = model.predict(x_test, batch_size=batch_size)
predictions = tf.nn.softmax(predictions)

print(predictions)
pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)


print('------------------------- Save and Load ----------------------------')

# -------1/ save the whole model---------------
# two possible formats: SavedModel |  HDF5

model.save("nn.h5")  # the latter format
model.save("nn")  # the former format

# -----load-----
load_model = tf.keras.models.load_model("nn.h5")
load_model.evaluate(x_test, y_test, verbose=2)


# ---------2/ save only weights-------------------
model.save_weights("nn_weights.h5")

# if you want to load weights, please to initialize model first:
# model = keras.Sequential([...])
model.load_weights("nn_weights.h5")


# ---------3/ save only architecture, to_json-------------------
json_string = model.to_json()

# save
with open("nn_model.json", "w") as f:
    f.write(json_string)

# load
with open("nn_model.json", "r") as f:
    loaded_json_string = f.read()

new_model = tf.keras.models.model_from_json(loaded_json_string)
print(new_model.summary())


























