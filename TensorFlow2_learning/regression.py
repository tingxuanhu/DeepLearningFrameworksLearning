import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

import tensorflow as tf

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
print(dataset.tail())  # last 5 rows
dataset = dataset.dropna()
print('-----------------------------------')

# Convert categorical 'Origin' data (i.e. 1, 2...) into one-hot data (i.e. 0,0,0,1 ....)
origin = dataset.pop('Origin')
# establish one-hot (create several labels)
dataset['USA'] = (origin == 1) * 1
dataset['Europe'] = (origin == 2) * 1
dataset['Japan'] = (origin == 3) * 1

print(dataset.head())

print(' --------------Split data into train and test ---------------------')

train_dataset = dataset.sample(frac=.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print(dataset.shape, train_dataset.shape, test_dataset.shape)
print(train_dataset.describe().transpose())

print('------------------------- split features from labels------------------------------')
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


# def plot(feature, x=None, y=None):
#     plt.figure(figsize=(10, 8))
#     plt.scatter(train_features[feature], train_labels, label='Data')
#     if x is not None and y is not None:
#         plt.plot(x, y, color='k', label='Predictions')
#     plt.xlabel(feature)
#     plt.ylabel('MPG')
#     plt.legend()
#
#
# print(plot('Horsepower'))

print('------------- Normalization -----------------')
# Normalize
print(train_dataset.describe().transpose()[['mean', 'std']])

# Normalization Layer
normalizer = tf.keras.layers.experimental.preprocessing.Normalization()

# adapt to the data
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# When the layer is called it returns the input data,
# with each feature independently normalized: (input-mean)/stddev
first = np.array(train_features[:1])
print('First example --> ', first)
print('Normalized:', normalizer(first).numpy())


print('----------------- Regression part -----------------------')
# 1/ Normalize the input horsepower

feature = 'Horsepower'
single_feature = np.array(train_features[feature])
# print(single_feature.shape, train_features.shape)  # (314,)  (314, 9)

single_feature_normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
single_feature_normalizer.adapt(single_feature)

# 2/ Apply a linear transformation (y = m*x+b) to produce 1 output using layers.Dense

# Sequential model
single_feature_model = tf.keras.models.Sequential([
    single_feature_normalizer,
    tf.keras.layers.Dense(units=1)  # units: dimensionality of the output space
])

print(single_feature_model.summary())


print('------------------ loss and optimizer ----------------')
loss = tf.keras.losses.MeanAbsoluteError()
optim = tf.keras.optimizers.Adam(lr=.1)

single_feature_model.compile(optimizer=optim, loss=loss)

history = single_feature_model.fit(train_features[feature], train_labels, epochs=100, verbose=1, validation_split=.2)
# history type : <tensorflow.python.keras.callbacks.History object at 0x132aa6550>
# print(history.history['loss'])


def plot_loss(history):
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


print(plot_loss(history))


print('---------- evaluation -----------------')
single_feature_model.evaluate(test_features[feature], test_labels, verbose=1)


print('---------- prediction -----------------')
range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)





























