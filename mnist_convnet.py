"""
Title: Simple MNIST convnet
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2015/06/19
Last modified: 2020/04/21
Updated by Matt Zucker: 2021/03/30
Updated by Patrick Kyaw: 2021/04/21
Description: A simple convnet that achieves ~99% test accuracy on MNIST.
Downloaded from: https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py
"""

########################################################################################
#Setup

from datetime import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

########################################################################################
#Prepare the data


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

########################################################################################
#Build the model

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        #layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        #layers.Dropout(0.1)
        layers.Dropout(0.5),  #HYPER PARAMETER TO BE CHANGED TO A LOWER RATE
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

########################################################################################
#Train the model


batch_size = 128
epochs = 15

train_begin = datetime.now()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

train_end = datetime.now()

########################################################################################
#Evaluate the trained model

score = model.evaluate(x_test, y_test, verbose=0)

test_end = datetime.now()

print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("Training time: ", train_end - train_begin)
print("Testing time: ", test_end - train_end)

########################################################################################
#Save the trained model so we can load it later (added by MZ 3/2021)

model.save('mnist_convnet_model', save_format='tf')
print('wrote mnist_convnet_model')
