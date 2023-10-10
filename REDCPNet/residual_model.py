import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, input, output, strides=1,dilation=1,padding=1,use_1x1conv=False,dil=False,flag=False):
        super(ResidualBlock, self).__init__()
        if dil and flag:
            dilation=5
            padding=5
        if dil and flag==False:
            dilation=2
            padding=2
        self.conv1 = nn.Conv2d(input, output, kernel_size=3,stride=strides,padding=padding,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(output)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output, output, kernel_size=3,padding=padding,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(output)
        if use_1x1conv==True:
            self.conv3=nn.Conv2d(input,output,1,stride=strides)
        else:
            self.conv3=None
        self.stride = strides

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))

        out=self.bn2(self.conv2(out))
        
        if self.conv3:
            x=self.conv3(x)

        out += x

        return F.relu(out)

def resnet_block(input,output,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if first_block:
            blk.append(ResidualBlock(input, output,use_1x1conv=True,strides=2))
        else:
            blk.append(ResidualBlock(output,output))
    return blk
