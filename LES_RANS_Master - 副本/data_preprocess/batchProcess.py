import data_preprocess.CSVToPicture
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

file_name = 'kcs-bare-rans-csv'
data_path = 'C:\\Users\\yyh\\Desktop\\LES与RANS\\LES_RANS数据\\RANS\\kcs-bare-rans\\'


for i in range(5,7):
    data_path_i = data_path + file_name + str(i) +'\\'
    for file in os.listdir(data_path_i):
        if os.path.isfile(os.path.join(data_path_i, file)) == True:
            oldname, suffx = os.path.splitext(file)  # oldname是文件名， suffx是文件类型
            # print(oldname, suffx)
            print(data_path_i)
            les_csvs = data_path_i + oldname + suffx
            save_path = 'D:\\c盘转移\\LES_RNS图像\\s=10\\RANS\\kcs-bare-rans\\' + file_name + str(i) + '\\' + oldname + '.png'
            # print(les_csvs, save_path)
            data_preprocess.CSVToPicture.CSV_to_pic(les_csvs, 15, save_path)

# for file in os.listdir(data_path):
#     if os.path.isfile(os.path.join(data_path, file)) == True:
#         oldname,suffx = os.path.splitext(file)          #oldname是文件名， suffx是文件类型
#         # print(oldname, suffx)
#         les_csvs = "C:\\Users\\yyh\\Desktop\\LES与RANS\\LES_RANS数据\\kcs-bare-les-csv2\\kcs-bare-les-csv2\\" + oldname + suffx
#         save_path = 'C:\\Users\\yyh\\Desktop\\深度学习\\s=1\\' + oldname +'.png'
#         # print(les_csvs, save_path)
#         data_preprocess.CSVToPicture.CSV_to_pic(les_csvs, 1, save_path)










