import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

le = preprocessing.LabelEncoder()

y_train = np.load("actualvalues.npy")
X_train = np.load("rawpixeldata.npy")

# print(np.unique(y_train))

le.fit(y_train)
y_train = le.transform(y_train)
# le.inverse_transform(y_train)

df = pd.DataFrame(X_train)
print(df.head())

img = df.iloc[1000]
img = img.values
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.show()