import os

path = "C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE"
 
# to store files in a list
list = []
 
# dirs=directories
import os.path
dirs = [d for d in os.listdir("C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\2006\\2006") if os.path.isdir(os.path.join("C:\\Users\milug\\Mihir's Important Stuff\\Mihir's Visual Studio\\NHSEE\\2006\\2006", d))]
print(dirs)