import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from keras.models import load_model
import numpy as np

model = load_model('bcnn2014.h5')
file = 'IFCB1_2006_158_000036_01314 copy.png'
file_pixel_data = []

oimg = cv2.imread(file, 0)
img = cv2.resize(oimg,(28,28))
for i in range (img.shape[0]):
        for j in range (img.shape[1]):
                k = img[i][j]
                k = k/255
                file_pixel_data.append(k)

formated_data = file_pixel_data.reshape(-1,28,28,1)
prediction = model.predict(formated_data)
predarray = np.array(prediction[0])