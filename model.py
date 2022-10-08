import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns


le = preprocessing.LabelEncoder()

y_train = np.load("actualvalues.npy")
X_train = np.load("rawpixeldata.npy")

print(len(y_train))
print(len(X_train))

X_train = X_train[: len(X_train) - 75000]
y_train = y_train[: len(y_train) - 75000]

X_train, y_train = shuffle(X_train, y_train)

print(y_train)
# print(np.unique(y_train))

le.fit(y_train)
y_train = le.transform(y_train)
# le.inverse_transform(y_train)

df = pd.DataFrame(X_train)
print(df.head())

# Reshape
# X_train = X_train.values.reshape(-1,28,28,1)
# test = test.values.reshape(-1,28,28,1)
# print("x_train shape: ",X_train.shape)
# print("test shape: ",test.shape)