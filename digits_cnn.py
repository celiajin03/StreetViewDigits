import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import scipy.io

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


# ------------ SET SEED ------------
seed_value= 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)


# ------------ READ DATA ------------
train = scipy.io.loadmat('dataset/SVHN/format2/train_32x32.mat')
test = scipy.io.loadmat('dataset/SVHN/format2/test_32x32.mat')

X_train = train['X']
y_train = train['y']
X_test = test['X']
y_test = test['y']


# ------------ PREPROCESS ------------
# Move axes of array X (last to first)
X_train = np.moveaxis(X_train, -1, 0)
X_test = np.moveaxis(X_test, -1, 0)

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Convert class vectors to binary class matrices
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

print("X_train.shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", X_test.shape)



"""
## Build the model
"""
# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.4),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()



"""
## Train the model
"""
batch_size = 128
epochs = 40

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard_callback])



"""
## Evaluate the trained model
"""
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get predictions and apply inverse transformation to the labels
y_pred = model.predict(X_test)
y_pred = enc.inverse_transform(y_pred)
y_test = enc.inverse_transform(y_test)

# Plot the confusion matrix
matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(matrix, annot=True, cmap='Blues', fmt='d', ax=ax, linewidths=2, xticklabels=range(1,11), yticklabels=range(1,11))
plt.title('Confusion Matrix for Test Dataset')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

