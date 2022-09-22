import torch
import torch.nn as nn
import torch.nn.functional as F



class basic_block(nn.Module):
    """基本残差块,由两层卷积构成"""
    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        """

        :param in_planes: 输入通道
        :param planes:  输出通道
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        """
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),nn.BatchNorm2d(planes))
        else:
            self.downsample = nn.Sequential()
    def forward(self,inx):
        x = self.relu(self.bn1(self.conv1(inx)))
        x = self.bn2(self.conv2(x))
        out = x+self.downsample(inx)
        return F.relu(out)
