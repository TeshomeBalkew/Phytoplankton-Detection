import os
import os.path
dirs = [d for d in os.listdir("C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006") if os.path.isdir(os.path.join("C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006", d))]

numfiles = 0
dir_size = 0
for folder in dirs:
    direct = "C:\\Users\\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\Phytoplankton Detection (10)\\2006\\2006\\" + folder
    direct = str(direct)
    for (path, dirs, file1) in os.walk(direct):
        for file in file1:
            filename = os.path.join(path, file)
            dir_size += os.path.getsize(filename)
            numfiles += 1

    if numfiles > 500:
        print("Folder", folder, "Size:", numfiles)
    # if dir_size == 0:
    #     print(folder + " -----------------------------------------Empty")
    # elif numfiles > 2000:
    #     print("\n")
    #     print("Folder ", folder, "Size: ", numfiles)
    #     print("\n")
    # else:
    #     print("Folder ", folder, "Size: ", numfiles)

    dir_size = 0
    numfiles = 0
