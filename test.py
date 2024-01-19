import keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, losses, optimizers, metrics

# Input: X - tensor will be of size (1000, 8)
# Output: Y - tensor will be of size (1000, 1)

# Import data from train.csv
data = pd.read_csv("dataset/train.csv", sep=",")

# Data preparation
tensor_data = tf.cast(tf.constant(data), dtype=tf.float32)
tensor_data = tf.random.shuffle(tensor_data)
# print(tensor_data[:4])

# We will get input for model.
X = tensor_data[:,3:-1]
# print(X.shape)

# And output for model.
Y = tf.expand_dims(tensor_data[:, -1], axis=-1)

# Constants for model.
_TRAIN_RATIO:      float = 0.8
_TEST_RATION:      float = 0.9
_DATASET_SIZE:     int   = len(X)


# Training data for the model.
X_train = X[:int(_DATASET_SIZE*_TRAIN_RATIO)]
Y_train = Y[:int(_DATASET_SIZE*_TRAIN_RATIO)]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)


# Validation data.
X_valid = X[int(_DATASET_SIZE*_TRAIN_RATIO):int(_DATASET_SIZE * _TEST_RATION)]
Y_valid = Y[int(_DATASET_SIZE*_TRAIN_RATIO):int(_DATASET_SIZE * _TEST_RATION)]

validation_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
validation_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

# Test data.
X_test = X[int(_DATASET_SIZE * _TEST_RATION):]
Y_test = Y[int(_DATASET_SIZE * _TEST_RATION):]

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)


# We need to normalize the data.
normalizer = layers.Normalization()

normalizer.adapt(X_train)
normalizer(X_train)[:5]

# Model.
model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(8, )))
model.add(normalizer)
# Hidden layers.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
# Output layer.
model.add(layers.Dense(1))
model.summary()

# Compile the model.
# model.compile(
#     optimizer=optimizers.Adam(learning_rate=0.1),
#     loss=losses.MeanAbsoluteError(),
#     metrics=metrics.RootMeanSquaredError()
# )

# Train the model.
# history = model.fit(train_dataset,
#         validation_data=validation_dataset, 
#         epochs=100, 
#         verbose=1)

# To see loss/epoch
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'])
# plt.show()

# print(model.predict(tf.expand_dims(X_test[0], axis=0)))
# print(Y_test[0])
