import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.utils import class_weight
import time

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

start = time.time()

le = preprocessing.LabelEncoder()

y_train = np.load("actval2014.npy")
X_train = np.load("rpd2014.npy")

X_train = X_train[: len(X_train)]
y_train = y_train[: len(y_train)]


cls_wgts = class_weight.compute_class_weight('balanced',
                                             sorted(np.unique(y_train)),
                                             y_train)
# dict mapping
cls_wgts = {i : cls_wgts[i] for i, label in enumerate(sorted(np.unique(y_train)))}


X_train, y_train = shuffle(X_train, y_train)



# Reshape
X_train = X_train.reshape(-1,28,28,1)

le.fit(y_train)
y_train = le.transform(y_train)

y_train = to_categorical(y_train, num_classes = 93)

dlen = len(X_train)
trainlen = math.ceil((dlen*90)/100)

#traindata
train_X = X_train[: dlen-trainlen]

#traintarget
train_y = y_train[: dlen-trainlen]

#testdata
test_X = X_train[dlen-trainlen :]

#testtarget
test_y = y_train[dlen-trainlen :]

print("train_x shape",train_X.shape)
print("train_y shape",train_y.shape)
print("test_X shape",test_X.shape)
print("test_y shape",test_y.shape)

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(93, activation = "softmax"))

# Compile the model
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 100  # for better result increase the epochs
batch_size = 128

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# Fit the model
history = model.fit(datagen.flow(train_X,train_y, batch_size=batch_size),
                              epochs = epochs, validation_data = (test_X,test_y), steps_per_epoch=train_X.shape[0] // batch_size, class_weight=cls_wgts)

model.save('bcnn2014.h5')

# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('lossmodelbc.png')

end = time.time()

print('Time Taken: ', end-start)