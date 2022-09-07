import numpy as np

load_npy = np.load("../data/a.npy")
a = 0
for j in range(2885):
    for i in range(2917):
        if load_npy[j][i][0] != 0:
            # print(load_npy[j][i])
            print(j, i, load_npy[j][i])
            a = a+1
print(a)
# print(load_npy)