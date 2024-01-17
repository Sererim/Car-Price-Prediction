import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers

# Input: X - tensor will be of size (1000, 8)
# Output: Y - tensor will be of size (1000, 1)

# Import data from train.csv
data = pd.read_csv("train.csv", sep=",")

# sns.pairplot(data[['years', 'km', 'rating', 'condition', 'economy', 'top speed',
#                    'hp', 'torque', 'current price']], diag_kind='kde')

# plt.show()

# Data preparation
tensor_data = tf.cast(tf.constant(data), dtype=tf.float32)
tensor_data = tf.random.shuffle(tensor_data)
# print(tensor_data[:4])

# We will get input for model.
X = tensor_data[:,3:-1]
# print(X.shape)

# And output for model.
Y = tf.expand_dims(tensor_data[:, -1], axis=-1)

# We need to normalize the data.
normalizer = layers.Normalization()

X = normalizer.adapt(X)

