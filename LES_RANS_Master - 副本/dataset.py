from torch.utils.data import Dataset
from utils import *

# def trans_list_to_tensor(data):
#     x = np.array(data)
#     x.dtype='float32'
#     return torch.from_numpy(x)
#

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.list = make_list(self.path + '\\xlsRANS')

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        segment_name = self.list[index][3]
        image_path = os.path.join(self.path, 'imgs', segment_name+'.png')
        img1, img2, img3 = dataCut(image_open(image_path))
        # res = dataGenerate(img1, img2, img3)
        file_name = segment_name + '.csv'
        rans_file = self.list[index][0:3]
        lab_les_file = find_label_les_data(self.path, file_name, rans_file)
        return img1, img2, img3, torch.from_numpy(np.array(rans_file)).float(), torch.from_numpy(np.array(lab_les_file)).float()
        # rans_file = np.array(rans_file, dtype='float64')
        # # rans_file.dtype= np.float32
        # # print(rans_file)
        # return img1, img2, img3torch.from_numpy(rans_file), np.array(lab_les_file)



if __name__ == '__main__':

    res = MyDataset('D:\\c盘转移\\LES_RANS_Master\\data')
    print(type(res[0][0]))
    print(res[0][1].shape)
