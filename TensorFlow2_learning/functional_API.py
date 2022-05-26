import tensorflow as tf

# model: Sequential: one input, one output
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
])

print(model.summary())

# ------------------ functional API --------------------

#  a           a             a     b          a
#  |           |              \    /        /   \
#  b           b                c          b     c
#  |          /  \              |          \     /
#  c         c    d             d             d

# create model with functional API
# Advantages:
#   - Models with multiple inputs and outputs
#   - Shared layers
#   - Extract and reuse nodes in the graph of layers
#   - Model are callable like layers (put model into sequential)

inputs = tf.keras.Input(shape=(28, 28))
flatten = tf.keras.layers.Flatten()
dense1 = tf.keras.layers.Dense(128, activation='relu')
dense2 = tf.keras.layers.Dense(10)
dense3 = tf.keras.layers.Dense(1)

x = flatten(inputs)
x = dense1(x)
x = dense2(x)
outputs = dense3(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='functional_model')
print(model.summary())


# convert functional to sequential model
# only works if the layers graph is linear.
new_model = tf.keras.models.Sequential()
for layer in model.layers:
    new_model.add(layer)

# convert sequential to functional
inputs = tf.keras.Input(shape=(28, 28))
x = new_model.layers[0](inputs)
for layer in new_model.layers[1:]:
    x = layer(x)
outputs = x






































