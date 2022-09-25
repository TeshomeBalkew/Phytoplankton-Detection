import os

path = "C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE"
 
# to store files in a list
list = []
 
# dirs=directories
import os.path
dirs = [d for d in os.listdir("C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\2006\\2006") if os.path.isdir(os.path.join("C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\2006\\2006", d))]

readable_sizes = {'B': 1,
                'KB': float(1) / 1024,
                'MB': float(1) / (1024 * 1024),
                'GB': float(1) / (1024 * 1024 * 1024)
                    }

dir_size = 0
for folder in dirs:
    direct = "C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\2006\\2006\\" + folder
    direct = str(direct)
    for (path, dirs, file1) in os.walk(direct):
        for file in file1:
            filename = os.path.join(path, file)
            dir_size += os.path.getsize(filename)

    if dir_size == 0:
        print("Folder ", folder, "Empty")
    else:
        print("Folder ", folder, "Size: ", dir_size)

    dir_size = 0
