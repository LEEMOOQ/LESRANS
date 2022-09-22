"""resnet18+全连接"""
from block import *
# from utils import *

class Net(nn.Module):
    def __init__(self, basicBlock, blockNums=[2, 2, 2, 2]):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(kernel_size=(1, 1), in_channels=3, out_channels=1)
        self.conv1_2 = nn.Conv2d(kernel_size=(1, 1), in_channels=3, out_channels=1)
        self.conv1_3 = nn.Conv2d(kernel_size=(1, 1), in_channels=3, out_channels=1)

        self.in_planes = 64
        #输入层
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(basicBlock, blockNums[0], 64, 1)
        self.layer2 = self._make_layers(basicBlock, blockNums[1], 128, 2)
        self.layer3 = self._make_layers(basicBlock, blockNums[2], 256, 2)
        self.layer4 = self._make_layers(basicBlock, blockNums[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.f1 = nn.Linear(515, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, 3)

    def _make_layers(self,basicBlock,blockNum,plane,stride):
        """
        :param basicBlock: 基本残差块类
        :param blockNum: 当前层包含基本残差块的数目,resnet18每层均为2
        :param plane: 输出通道数
        :param stride: 卷积步长
        :return:
        """
        layers=[]
        for i in range(blockNum):
            if i == 0:
                layer = basicBlock(self.in_planes, plane, 3, stride=stride)
            else:
                layer = basicBlock(plane, plane, 3, stride=1)
            layers.append(layer)
        self.in_planes = plane
        return nn.Sequential(*layers)

    def forward(self, inx1, inx2, inx3, rans_loc):
        x1 = self.conv1_1(inx1)
        x2 = self.conv1_2(inx2)
        x3 = self.conv1_3(inx3)
        x = torch.cat((x1, x2, x3), dim=1)
        # print(x.shape)

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape, rans_loc.shape)
        x = torch.cat((x, rans_loc), dim=1)
        # print(x.shape)
        x = self.f1(x)
        x = self.f2(x)
        out = self.f3(x)
        return out
#
if __name__=="__main__":
    net = Net(basic_block, [2, 2, 2, 2])
    # print(resnet18)
    inx1 = torch.randn(4, 3, 256, 256)
    inx2 = torch.randn(4, 3, 256, 256)
    inx3 = torch.randn(4, 3, 256, 256)
    to = torch.randn(4,3)
    # print(inx.shape)
    outx = net(inx1, inx2, inx3, to)
    print(outx.shape)
