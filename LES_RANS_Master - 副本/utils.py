import os
import pandas as pd
import numpy as np
import torch

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

#list转tensor
#neg_list = []
# neg_numpy = np.array(neg_list)
# neg_tensor = torch.from_numpy(neg_numpy)

# def Keep_image_size_open(path, size=(256, 256)):
#     img = Image.open(path)
#     temp = max(img.size)
#     mask = Image.new('RGB', (temp, temp), (0, 0, 0))
#     mask.paste(img, (0, 0))
#     mask = mask.resize(size)
#     return mask

transform = transforms.Compose([
    transforms.ToTensor()
])

def image_open(path):
    img = Image.open(path)
    return img

def keep_image_size_open_rgb(img):
    mask = img.resize((256, 256), Image.BILINEAR)
    return mask

def dataCut(data):
    box1 = (40, 950, 490, 1300)
    box2 = (450, 500, 900, 850)
    box3 = (870, 45, 1310, 395)
    pic1 = keep_image_size_open_rgb(data.crop(box1)).convert("RGB")
    pic2 = keep_image_size_open_rgb(data.crop(box2)).convert("RGB")
    pic3 = keep_image_size_open_rgb(data.crop(box3)).convert("RGB")
    return transform(pic1), transform(pic2), transform(pic3)

    #左下图片：（35，880，545，1280）
    #中间图片：（435，480，915，880）
    #右上图片：（835，50，1315，450）

def dataGenerate(x, y, z):
    x1 = x[0]+x[1]+x[2]
    x1 = torch.unsqueeze(x1, 0)
    y1 = y[0]+y[1]+y[2]
    y1 = torch.unsqueeze(y1,0)
    z1 = z[0]+z[1]+z[2]
    z1 = torch.unsqueeze(z1,0)
    res = torch.cat((x1, y1, z1), axis=0)
    return res


def make_list(path):
    list = []
    #path是RANS的excel数据地址
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) == True:
    #         file = path
            oldname, suffx = os.path.splitext(file)  # oldname是文件名， suffx是文件类型
            df = pd.read_csv(os.path.join(path, file), usecols=[3, 4, 5], names=None)  # 读取xls各列，并不要列名
            x_mode = df.loc[:, 'X (m)'].mode()[0]
            df = df.drop(df[df.loc[:, 'X (m)'] != x_mode].index)
            df = df.reset_index(drop=True)
            row_list = df.values.tolist()
            for i in range(len(row_list)):
                row_list[i].append(oldname)
            if list == []:
                list = row_list
            else:
                list = list + row_list
    return list


def find_label_les_data(path, filename, rans_file):
    file_path = os.path.join(path, 'xlsLES', filename)
    df = pd.read_csv(file_path, usecols=[0, 1, 2, 3, 4, 5], names=None, dtype=float)  # 读取xls各列，并不要列名
    x_mode = df.loc[:, 'X (m)'].mode()[0]
    df = df.drop(df[df.loc[:, 'X (m)'] != x_mode].index)
    df = df.reset_index(drop=True)
    row_list = df.values.tolist()
    # print(row_list)
    for i in range(len(row_list)):
        if rans_file[0:3] == row_list[i][3:6]:
            # return torch.unsqueeze(torch.from_numpy(np.array(row_list[i][0:3], dtype=float)), dim=0)
            return row_list[i][0:3]

# xls_path_test = "test/"
xls_path = "D:\\c盘转移\\LES_RANS_Master\\data"     #XYZ_Internal_Table_table_1.000992e-01.csv
file_name = 'XYZ_Internal_Table_table_1.000992e-01.csv'
rans_file = [0, 0.140625, -0.237848450000001]
# path = 'D:\\c盘转移\\LES_RNS图像\\s=10\\RANS\\kcs-bare-rans\\kcs-bare-rans-csv1\\XYZ_Internal_Table_table_1.000992e-01.png'
# # img1, img2, img3 = dataCut(image_open(path))
# print(img1.dtype)
# res = dataGenerate(img1, img2, img3)
# print(res.size())
# plt.figure("Image")  # 图像窗口名称
# plt.imshow(img1)
# plt.show()
# df = make_list(xls_path)
# # for i in range(26528):
# #     df[i].append('test')
# print(df[0])
#
# les_data = find_label_les_data(xls_path, file_name, rans_file)
# print(les_data[0])
# for file in os.listdir(xls_path_test):
#     print(file)