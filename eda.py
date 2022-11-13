import pandas as pd
import cv2
import time
import numpy as np
import os
import os.path

start = time.time()
dirs = [d for d in os.listdir("C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2014\\2014") if os.path.isdir(os.path.join("C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2014\\2014", d))]
folderlist = []

numfiles = 0
dir_size = 0
for folder in dirs:
    direct = "C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2014\\2014\\" + folder
    direct = str(direct)
    for (path, dirs, file1) in os.walk(direct):
        for file in file1:
            filename = os.path.join(path, file)
            dir_size += os.path.getsize(filename)
            numfiles += 1
        if numfiles!=0 and folder!='mix':
                folderlist.append(folder)
    dir_size = 0
    numfiles = 0

pixeldatalist = []
actvalues = []

print(folderlist)

colname = []
for val in range(784):
        colname.append(val)
colname.append('label')
datan = pd.DataFrame(columns = colname)


for folder in folderlist:
    direct = "C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2014\\2014\\" + folder
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

np.save('rpd2014', xtrain)
np.save('actval2014', ytrain)

end = time.time()
print('Time Taken: ', end-start)