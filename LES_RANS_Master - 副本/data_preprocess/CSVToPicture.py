import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def CSV_to_pic(bare_les_csvs_path, s, save_path):

    bles_csv = pd.read_csv(bare_les_csvs_path)
    bles_csv = bles_csv.sort_values(by=['Y (m)', 'Z (m)']).reset_index()
    x_mode = bles_csv.loc[:, 'X (m)'].mode()[0]
    bles_csv = bles_csv.drop(bles_csv[bles_csv.loc[:, 'X (m)'] != x_mode].index)
    bles_csv = bles_csv.reset_index(drop=True)

    x_value = np.array(bles_csv.loc[:, 'X (m)'])
    y_value = np.array(bles_csv.loc[:, 'Y (m)'])
    z_value = np.array(bles_csv.loc[:, 'Z (m)'])

    d_value_1 = np.array(bles_csv.loc[:, 'Velocity in Cylindrical 1[Axial] (m/s)'])
    d_value_2 = np.array(bles_csv.loc[:, 'Velocity in Cylindrical 1[Tangential] (m/s)'])
    d_value_3 = np.array(bles_csv.loc[:, 'Velocity in Cylindrical 1[Radial] (m/s)'])

    # fig = plt.figure(figsize=(5, 5), dpi=500)
    # ax_1 = fig.add_subplot(111)
    # ax_1.set_xlabel('y', fontsize=14)
    # ax_1.set_ylabel('z', fontsize=14)
    # plt.tick_params(labelsize=10)
    # ax_1.scatter(x=[y_value, y_value + 1, y_value + 2], y=[z_value, z_value + 1, z_value + 2],
    #              c=[d_value_1, d_value_2, d_value_3], s = s, cmap='jet', marker='o')
    # fig.canvas.draw()
    # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.axis('off')  # 去坐标轴
    # plt.xticks([])  # 去 x 轴刻度
    # plt.yticks([])  # 去 y 轴刻度
    # plt.show()
    # # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=350)


if __name__ == '__main__':
    name = "XYZ_Internal_Table_table_3.859125e-02"
    bare_les_csvs = "D:\c盘转移\LES_RNS图像\xls数据\LES\kcs-bare-les\kcs-bare-les-csv1\\XYZ_Internal_Table_table_1.000992e-01.csv"
    # save_path = 'C:\\Users\\yyh\\Desktop\\深度学习\\s=10\\3_Radial\\test.png'
    save_path = 'D:\\c盘转移\\LES_RNS图像\\s=10\\LES\\kcs-bare-les\\kcs-bare-les-csv3\\'+ name +'.png'
    CSV_to_pic(bare_les_csvs, 15, save_path)     #1\15