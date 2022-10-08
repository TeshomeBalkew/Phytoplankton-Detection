import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


le = preprocessing.LabelEncoder()

y_train = np.load("actualvalues.npy")
X_train = np.load("rawpixeldata.npy")

X_train = X_train[: len(X_train) - 75000]
y_train = y_train[: len(y_train) - 75000]

X_train, y_train = shuffle(X_train, y_train)



# Reshape
X_train = X_train.reshape(-1,28,28,1)

le.fit(y_train)
y_train = le.transform(y_train)

y_train = to_categorical(y_train, num_classes = 7)

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