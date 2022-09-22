from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import *
from net import *
from block import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/net.pth'
data_path = 'data'
test_data_path = 'testData'
epoch_all = 30
#
#26562
data_loader = DataLoader(MyDataset(data_path), batch_size=100, shuffle=False)
test_loader = DataLoader(MyDataset(test_data_path), batch_size=100, shuffle=False)

#训练函数
def train(data_loader, net, loss_fn, opt, epoch):
    all_train = 0
    net.train()
    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for i, (img1, img2, img3, rans_loc, les_v) in loop:
        img1, img2, img3, rans_loc, les_v = img1.to(device), img2.to(device), img3.to(device), rans_loc.to(device), les_v.to(device)
        out_v = net(img1, img2, img3, rans_loc)
        train_loss = loss_fn(out_v, les_v)
        # print(train_loss)
        opt.zero_grad()
        train_loss.backward()
        opt.step()

        all_train += train_loss.item()
        loop.set_description(f'Epoch [{epoch}/{epoch_all}]')
        loop.set_postfix(loss=train_loss.item())

        if i % 300 == 1:
            torch.save(net.state_dict(), weight_path)

    print(f'第{epoch}轮：train_loss===>>{all_train}')
    return all_train


#验证函数
def val(test_loader, model, loss_fn, epoch):
    #将模型转化为验证模式
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for i, (img1, img2, img3, rans_loc, les_v) in enumerate(tqdm(test_loader, leave=False)):
            img1, img2, img3, rans_loc, les_v = img1.to(device), img2.to(device), img3.to(device), rans_loc.to(device), les_v.to(device)
            out_v = model(img1, img2, img3, rans_loc)
            cur_loss = loss_fn(out_v, les_v)
            loss += cur_loss.item()
    val_loss = loss
    print(f'第{epoch}轮：test_loss====>>{val_loss}')
    return val_loss


if __name__ == '__main__':
    net = Net(basic_block).to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt = optim.Adam(net.parameters())
    # loss_fun = nn.CrossEntropyLoss()      #二分类或多分类，自带sigmoid
    loss_fn = nn.MSELoss()

    loss_num = []
    test_loss = []
    epoch = 0
    min_loss = 100

    print("开始训练！")
    while epoch < epoch_all:
        all_train = train(data_loader, net, loss_fn, opt, epoch)
        val_loss = val(test_loader, net, loss_fn, epoch)

        if min_loss>all_train:
            min_loss = all_train
            torch.save(net.state_dict(), 'params/best.pth')         #保存最优pth
        loss_num.append(all_train)
        test_loss.append(val_loss)
        epoch += 1


    # x_data = []
    for i in range(len(loss_num)):
        print(loss_num[i])
        print(test_loss[i])

    #     x_data.append(i)
    # plt.plot(x_data, loss_num, 'b*--', alpha=0.5, linewidth=1, label='train_loss')  # '
    # plt.plot(x_data, val_loss, 'g*--', alpha=0.5, linewidth=1, label='test_loss')  # '
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.savefig('./500epochs.png')
    # # plt.ylim(-1,1)#仅设置y轴坐标范围
    # plt.show()

