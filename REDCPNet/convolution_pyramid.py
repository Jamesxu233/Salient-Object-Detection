import torch
from torch import nn
from torch.nn import functional as F

class conpy(nn.Module):
    def __init__(self,input_channles,output1,output2,output3,input_size):
        super(conpy,self).__init__()
        #Branch 1,single 1x1 conv
        self.b1_1=nn.Conv2d(input_channles,output1,1)
        #Branch 2,contain 4 parallel sub-branches each contain 2 conv
        self.b2_s1_1=nn.Conv2d(input_channles,output2[0],kernel_size=(1,3),padding=1)
        self.b2_s1_2=nn.Conv2d(output2[0],output2[1],kernel_size=(3,1))
        self.b2_s2_1=nn.Conv2d(input_channles,output2[0],kernel_size=(1,5),padding=2)
        self.b2_s2_2=nn.Conv2d(output2[0],output2[1],kernel_size=(5,1))
        self.b2_s3_1=nn.Conv2d(input_channles,output2[0],kernel_size=(1,7),padding=3)
        self.b2_s3_2=nn.Conv2d(output2[0],output2[1],kernel_size=(7,1))
        self.b2_s4_1=nn.Conv2d(input_channles,output2[0],kernel_size=(1,9),padding=4)
        self.b2_s4_2=nn.Conv2d(output2[0],output2[1],kernel_size=(9,1))
        #Branch 3,contain a global pooling layer,a 1x1 conv 
        self.b3_1=nn.AdaptiveAvgPool2d((1,1))
        self.b3_2=nn.Conv2d(input_channles,output3,1)
        self.b3_3=nn.Upsample(input_size,mode='bilinear')
    def forward(self,x):
        b1=self.b1_1(x)
        b2_s1=self.b2_s1_2(self.b2_s1_1(x))
        b2_s2=self.b2_s2_2(self.b2_s2_1(x))
        b2_s3=self.b2_s3_2(self.b2_s3_1(x))
        b2_s4=self.b2_s4_2(self.b2_s4_1(x))
        b3=self.b3_3(self.b3_2(self.b3_1(x)))
        return F.relu(torch.cat((b1,b2_s1,b2_s2,b2_s3,b2_s4,b3),dim=1))


