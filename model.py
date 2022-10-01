# import matplotlib.pyplot as plt
# import seaborn as sns
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,confusion_matrix
# from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import cv2
import os
import os.path
pixeldatalist = []
dirs = ['Chaetoceros', 'Skeletonema', 'Thalassionema', 'Guinardia_delicatula', 'Leptocylindrus', 'Mesodinium_sp', 'mix']

numfiles = 0
dir_size = 0
for folder in dirs:
    direct = "C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006\\" + folder
    direct = str(direct)
    for (path, dirs, file1) in os.walk(direct):
        for file in file1:
                filename = os.path.join(path, file)
                nlist = []
                image = cv2.imread(filename)
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                analysisframe = cv2.resize(gray,(28,28))

                rows,cols = analysisframe.shape
                for i in range(rows):
                        for j in range(cols):
                                k = analysisframe[i,j]
                                nlist.append(k)
                nlist.append(folder)
                pixeldatalist.append(nlist)
                nlist = []

datan = pd.DataFrame(pixeldatalist)
colname = []
for val in range(784):
        colname.append(val)
colname.append('label')
datan.columns = colname

datan.to_csv('bacteria_img_data.csv', index = False)


# pixeldata = datan.values
# pixeldata = pixeldata / 255
# pixeldata = pixeldata.reshape(-1,28,28,1)













# #convert image data to model
# y_train = train_df['label']
# y_test = test_df['label']
# del train_df['label']
# del test_df['label']

# from sklearn.preprocessing import LabelBinarizer
# label_binarizer = LabelBinarizer()
# y_train = label_binarizer.fit_transform(y_train)
# y_test = label_binarizer.fit_transform(y_test)

# x_train = train_df.values
# x_test = test_df.values

# x_train = x_train / 255
# x_test = x_test / 255

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)

# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False)  # randomly flip images

# datagen.fit(x_train)

# learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

# model = Sequential()
# model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
# model.add(Flatten())
# model.add(Dense(units = 512 , activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(units = 24 , activation = 'softmax'))
# model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
# model.summary()

# history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 20 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

# model.save('smnist.h5')