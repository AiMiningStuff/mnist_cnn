"""
Source: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
"""

import numpy as np
from keras.models import Sequential  # this is a linear stack of nn layers
from keras.layers import Dense, Dropout, Flatten  # basic nn layers
from keras.layers import Conv2D, MaxPooling2D  # cnn specific layers
from keras.utils import np_utils  # utilities for transforming data
from keras.datasets import mnist  # the mnist dataset
from keras.models import model_from_json

np.random.seed(123)  # for reproducibility

(X_train, y_train), (X_test, y_test) = mnist.load_data()

from matplotlib import pyplot as plt

# strighten data
X_train[X_train > 0] = 255
X_test[X_test > 0] = 255

X_train_ex = X_train[0]

print(X_train_ex.shape)
X_train_ex = X_train_ex.reshape(784)
print(X_train_ex)
print(X_train_ex.shape)

X_train_ex = X_train_ex.reshape(28,28)

plt.imshow(X_train_ex, cmap='gray')

# reshape data so that we specify depth
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print(X_train.shape)

# normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
# putting the respective classes into the right boxes
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print(Y_train.shape)

# Declear sequential model
model = Sequential()

# add a convolutional input layer
# The step size is (1,1) by default, and it can be tuned using the 'subsample' 
# parameter
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28), \
                 data_format="channels_first"))

# add more layers
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# add a fully connected layer and then the output layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# fit model on data
model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)

# evaluate the model on test data
score = model.evaluate(X_test, Y_test, verbose=0)

# serialize model to JSON
model_json = model.to_json()
with open("model_struct.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")