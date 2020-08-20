
import os
import sys

for i in range(int(sys.argv[3])):
	os.system("wget "+sys.argv[1]+str(i+1)+".jpg -P "+sys.argv[2])