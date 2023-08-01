# -*- coding: utf-8 -*-
# Please run this script in "Pixelv2" directory
# Never run this script on Server

import os
import platform

split_char = "/"

sys = platform.system()
if sys == "Windows":
    split_char = "\\"
else:
    split_char = "/"

pwd = os.getcwd()
if pwd.split(split_char)[-1] != "Pixelv2":
    print("Please run this script in Pixelv2 directory")
    exit(1)

max_num = int(input("Delete all fingerprinting files with number larger than: "))

top_dir = ['cpu', 'gpu']

for cur_dir in top_dir:
    dir = os.path.join(pwd, cur_dir)
    # Get all files in Pixelv2 directory
    files = os.listdir(dir)
    # Get all directories in Pixelv2 directory
    sub_dirs = [f for f in files if os.path.isdir(os.path.join(dir, f))]
    print (sub_dirs)
    for sd in sub_dirs:
        fp_dir = os.path.join(dir, sd)
        files = os.listdir(fp_dir)
        for f in files:
            num = f.split('.')[0]
            if int(num) > max_num:
                os.remove(os.path.join(fp_dir, f))
                print ("Delete file: ", os.path.join(fp_dir, f))
