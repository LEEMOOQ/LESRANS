import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUM_X_0 = 3
NUM_Y = 4
NUM_Z = 5


def xls__to_npy():
    df = pd.read_excel("C:\\Users\\yyh\Desktop\\LES与RANS\\LES_RANS数据\\kcs-bare-les-csv2\\kcs-bare-les-csv2\\test.xls", usecols=[0, 1, 2, 3, 4, 5],
                       names=None)  # 读取xls各列，并不要列名
    df_li = df.values.tolist()
    i1 = 0
    # 去除x ！= 0 项
    while i1 < len(df_li):
        if df_li[i1][NUM_X_0] != 0:
            del df_li[i1]
        else:
            i1 = i1+1

    # print(df_li)
    # nu_li = np.array(df_li)  #转化为numpy

    # 定义 x  y  nu变量
    x = [df_li[0][NUM_Y]]   #存储y轴坐标
    y = [df_li[0][NUM_Z]]   #存储z轴坐标
    nu1 = df_li[0][0:3]
    nu = [np.array(nu1)]    #存储所有点信息[轴向 切线 径向]

    # 用for循环将文件中的值赋值给x，y, nu
    for i in range(1, len(df_li)):
        x.append(df_li[i][NUM_Y])
        y.append(df_li[i][NUM_Z])
        nu1 = df_li[i][0:3]
        nu.append(np.array(nu1))

    # print(nu)
    nu_li = np.array(nu, dtype=object)  # 转化为numpy
    # print(nu_li)

    # 对x y从大到小排序
    x1 = sorted(x)
    y1 = sorted(y)

    num = x1[0]
    x2 = [num]

    # 找到y轴z轴所有不重复点坐标并存储进x2、y2
    for i in range(len(x1)):
        if num != x1[i]:
            x2.append(x1[i])
            num = x1[i]
    num = y1[0]
    y2 = [num]
    for i in range(len(y1)):
        if num != y1[i]:
            y2.append(y1[i])
            num = y1[i]

    #创建存储npy数据的三维数组
    ans = np.zeros((len(x2), len(y2), 3))
    # print(ans)

    print(len(x2), len(y2))
    # # 从两张向量中推测相关位置
    for i in range(0, len(x)):
        a = 0
        b = 0
        while x[i] > x2[a]:
            a = a+1
        while y[i] > y2[b]:
            b = b+1
        ans[a][b] = nu_li[i]

    #保存为npy文件
    np.save('../data/a.npy', ans)


if __name__ == '__main__':
    xls__to_npy()











