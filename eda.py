import pandas as pd
import cv2
import os
import os.path
import time
import numpy as np

pixeldatalist = []
actvalues = []
dirs = ['Chaetoceros', 'Skeletonema', 'Thalassionema', 'Guinardia_delicatula', 'Leptocylindrus', 'Mesodinium_sp', 'mix']

numfiles = 0
dir_size = 0

colname = []
for val in range(784):
        colname.append(val)
colname.append('label')
datan = pd.DataFrame(columns = colname)


for folder in dirs:
    direct = "C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006\\" + folder
    direct = str(direct)
    for (path, dirs, file1) in os.walk(direct):
        for file in file1:
                filename = os.path.join(path, file)
                nlist = []
                randval = 0
                image = cv2.imread(filename, 0)
                try:
                        analysisframe = cv2.resize(image,(28,28))
                except:
                        print(filename)
                        randval = 1


                if randval == 1:
                        randval = 0
                        continue
                else:
                        for i in range (analysisframe.shape[0]):
                                for j in range (analysisframe.shape[1]):
                                        k = analysisframe[i][j]
                                        k = k/255
                                        nlist.append(k)
                        actvalues.append(folder)
                        pixeldatalist.append(nlist)
                        nlist = []


xtrain = np.array(pixeldatalist)
ytrain = np.array(actvalues)

print(xtrain)
print(ytrain)

np.save('rawpixeldata', xtrain)
np.save('actualvalues', ytrain)


# datan.to_csv('bacteria_img_data.csv', index = False)
# import os
# import os.path
# dirs = [d for d in os.listdir("C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006") if os.path.isdir(os.path.join("C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006", d))]

# numfiles = 0
# dir_size = 0
# for folder in dirs:
#     direct = "C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006\\" + folder
#     direct = str(direct)
#     for (path, dirs, file1) in os.walk(direct):
#         for file in file1:
#             filename = os.path.join(path, file)
#             dir_size += os.path.getsize(filename)
#             numfiles += 1

#     if numfiles > 500:
#         print("Folder", folder, "Size:", numfiles)
#     # if dir_size == 0:
#     #     print(folder + " -----------------------------------------Empty")
#     # elif numfiles > 2000:
#     #     print("\n")
#     #     print("Folder ", folder, "Size: ", numfiles)
#     #     print("\n")
#     # else:
#     #     print("Folder ", folder, "Size: ", numfiles)

#     dir_size = 0
#     numfiles = 0