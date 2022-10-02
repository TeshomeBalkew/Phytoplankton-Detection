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

import cv2

im = cv2.imread("C:\\Users\\milug\\Downloads\\63838cnntoday.png")

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray,(28,28))


for i in range (gray.shape[0]): #traverses through height of the image
    for j in range (gray.shape[1]): #traverses through width of the image
        print(gray[i][j])