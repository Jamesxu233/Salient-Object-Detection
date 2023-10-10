import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from residual_model import*
from convolution_pyramid import*


class REDCPNet(nn.Module):
    def __init__(self,input_channels):
        super(REDCPNet,self).__init__()
     
        ##----------Encoder----------

        self.inconv=nn.Conv2d(input_channels,64,3,padding=1)
        self.inbn=nn.BatchNorm2d(64)
        self.inrelu=nn.ReLU(True)

        #stage 1
        self.resb1_1=ResidualBlock(64,64)
        self.resb1_2=ResidualBlock(64,64,dil=True)
        self.resb1_3=ResidualBlock(64,64,dil=True,flag=True)

        self.conpy1=conpy(64,64,(64,64),64,224)

        self.tfr1=nn.Conv2d(384,64,1)

        
        #stage 2
        self.resb2_1=ResidualBlock(64,128,use_1x1conv=True,strides=2)
        self.resb2_2=ResidualBlock(128,128,dil=True)
        self.resb2_3=ResidualBlock(128,128,dil=True,flag=True)

        self.conpy2=conpy(128,128,(128,128),128,112)
        
        self.tfr2=nn.Conv2d(768,128,1)
        
        
        #stage 3
        self.resb3_1=ResidualBlock(128,256,use_1x1conv=True,strides=2)
        self.resb3_2=ResidualBlock(256,256,dil=True)
        self.resb3_3=ResidualBlock(256,256,dil=True,flag=True)
        
        self.conpy3=conpy(256,256,(256,256),256,56)

        self.trf3=nn.Conv2d(1536,256,1)


        
        #stage 4
        self.resb4_1=ResidualBlock(256,512,use_1x1conv=True,strides=2)
        self.resb4_2=ResidualBlock(512,512,dil=True)
        self.resb4_3=ResidualBlock(512,512,dil=True,flag=True)

        self.conpy4=conpy(512,512,(512,512),512,28)

        self.trf4=nn.Conv2d(3072,512,1)


        #stage 5
        self.resb5_1=ResidualBlock(512,512,use_1x1conv=True,strides=2)
        self.resb5_2=ResidualBlock(512,512,dil=True)
        self.resb5_3=ResidualBlock(512,512,dil=True,flag=True)

        self.conpy5=conpy(512,512,(512,512),512,14)

        self.trf5=nn.Conv2d(3072,512,1)


        #stage 6
        self.resb6_1=ResidualBlock(512,512,use_1x1conv=True,strides=2)
        self.resb6_2=ResidualBlock(512,512,dil=True)
        self.resb6_3=ResidualBlock(512,512,dil=True,flag=True)

        ##----------Bridge----------
        self.convbg_1=nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_1=nn.BatchNorm2d(512)
        self.relubg_1=nn.ReLU(inplace=True)
        self.convbg_m=nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m=nn.BatchNorm2d(512)
        self.relubg_m=nn.ReLU(inplace=True)
        self.convbg_2=nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2=nn.BatchNorm2d(512)
        self.relubg_2=nn.ReLU(inplace=True)
    
    ##----------Decoder----------

        #stage 6d
        self.conv6d_1=nn.Conv2d(1024,512,3,padding=1)
        self.bn6d_1=nn.BatchNorm2d(512)
        self.relu6d_1=nn.ReLU(inplace=True)
        
        self.conv6d_m=nn.Conv2d(512,512,3,dilation=2,padding=2)
        self.bn6d_m=nn.BatchNorm2d(512)
        self.relu6d_m=nn.ReLU(inplace=True)
        
        self.conv6d_2=nn.Conv2d(512,512,3,dilation=2,padding=2)
        self.bn6d_2=nn.BatchNorm2d(512)
        self.relu6d_2=nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1=nn.Conv2d(1024,512,3,padding=1)
        self.bn5d_1=nn.BatchNorm2d(512)
        self.relu5d_1=nn.ReLU(inplace=True)

        self.conv5d_m=nn.Conv2d(512,512,3,padding=1)
        self.bn5d_m=nn.BatchNorm2d(512)
        self.relu5d_m=nn.ReLU(inplace=True)

        self.conv5d_2=nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2=nn.BatchNorm2d(512)
        self.relu5d_2=nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1=nn.Conv2d(1024,512,3,padding=1)
        self.bn4d_1=nn.BatchNorm2d(512)
        self.relu4d_1=nn.ReLU(inplace=True)

        self.conv4d_m=nn.Conv2d(512,512,3,padding=1)
        self.bn4d_m=nn.BatchNorm2d(512)
        self.relu4d_m=nn.ReLU(inplace=True)

        self.conv4d_2=nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2=nn.BatchNorm2d(256)
        self.relu4d_2=nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1=nn.Conv2d(512,256,3,padding=1)
        self.bn3d_1=nn.BatchNorm2d(256)
        self.relu3d_1=nn.ReLU(inplace=True)

        self.conv3d_m=nn.Conv2d(256,256,3,padding=1)
        self.bn3d_m=nn.BatchNorm2d(256)
        self.relu3d_m=nn.ReLU(inplace=True)

        self.conv3d_2=nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2=nn.BatchNorm2d(128)
        self.relu3d_2=nn.ReLU(inplace=True)

        #stage 2d
        self.conv2d_1=nn.Conv2d(256,128,3,padding=1)
        self.bn2d_1=nn.BatchNorm2d(128)
        self.relu2d_1=nn.ReLU(inplace=True)

        self.conv2d_m=nn.Conv2d(128,128,3,padding=1)
        self.bn2d_m=nn.BatchNorm2d(128)
        self.relu2d_m=nn.ReLU(inplace=True)

        self.conv2d_2=nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2=nn.BatchNorm2d(64)
        self.relu2d_2=nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1=nn.Conv2d(128,64,3,padding=1)
        self.bn1d_1=nn.BatchNorm2d(64)
        self.relu1d_1=nn.ReLU(inplace=True)

        self.conv1d_m=nn.Conv2d(64,64,3,padding=1)
        self.bn1d_m=nn.BatchNorm2d(64)
        self.relu1d_m=nn.ReLU(inplace=True)

        self.conv1d_2=nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2=nn.BatchNorm2d(64)
        self.relu1d_2=nn.ReLU(inplace=True)

        ##----------Bilinear Upsampling----------
        self.upsm6=nn.Upsample(scale_factor=32,mode='bilinear')
        self.upsm5=nn.Upsample(scale_factor=16,mode='bilinear')
        self.upsm4=nn.Upsample(scale_factor=8,mode='bilinear')
        self.upsm3=nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsm2=nn.Upsample(scale_factor=2, mode='bilinear')

        ##----------Output----------
        self.outconvb=nn.Conv2d(512,1,3,padding=1)
        self.outconv6=nn.Conv2d(512,1,3,padding=1)
        self.outconv5=nn.Conv2d(512,1,3,padding=1)
        self.outconv4=nn.Conv2d(256,1,3,padding=1)
        self.outconv3=nn.Conv2d(128,1,3,padding=1)
        self.outconv2=nn.Conv2d(64,1,3,padding=1)
        self.outconv1=nn.Conv2d(64,1,3,padding=1)
    
    def forward(self,x):
        hx=x

        ##----------Encoder----------
        hx=self.inconv(hx)
        hx=self.inbn(hx)
        hx=self.inrelu(hx)

        hx=self.resb1_1(hx)
        hx=self.resb1_2(hx)
        hx=self.resb1_3(hx)
        h1=self.conpy1(hx)
        h1=self.tfr1(h1)


        hx=self.resb2_1(h1)
        hx=self.resb2_2(hx)
        hx=self.resb2_3(hx)
        h2=self.conpy2(hx)
        h2=self.tfr2(h2)


        hx=self.resb3_1(h2)
        hx=self.resb3_2(hx)
        hx=self.resb3_3(hx)
        h3=self.conpy3(hx)
        h3=self.trf3(h3)


        hx=self.resb4_1(h3)
        hx=self.resb4_2(hx)
        hx=self.resb4_3(hx)
        h4=self.conpy4(hx)
        h4=self.trf4(h4)


        hx=self.resb5_1(h4)
        hx=self.resb5_2(hx)
        hx=self.resb5_3(hx)
        h5=self.conpy5(hx)
        h5=self.trf5(h5)

        hx=self.resb6_1(hx)
        hx=self.resb6_2(hx)
        h6=self.resb6_3(hx)

        ## -------------Bridge-------------
        hx=self.relubg_1(self.bnbg_1(self.convbg_1(h6)))
        hx=self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg=self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx=self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx=self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6=self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx=self.upsm2(hd6)

        hx=self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx=self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5=self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx=self.upsm2(hd5)

        hx=self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hx=self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4=self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx=self.upsm2(hd4)

        hx=self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx=self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3=self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx=self.upsm2(hd3)

        hx=self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx=self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2=self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx=self.upsm2(hd2)

        hx=self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx=self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1=self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db=self.outconvb(hbg)
        db=self.upsm6(db)

        d6=self.outconv6(hd6)
        d6=self.upsm6(d6)

        d5=self.outconv5(hd5)
        d5=self.upsm5(d5)

        d4=self.outconv4(hd4)
        d4=self.upsm4(d4)

        d3=self.outconv3(hd3)
        d3=self.upsm3(d3)

        d2=self.outconv2(hd2)
        d2=self.upsm2(d2)

        d1=self.outconv1(hd1)

        return F.sigmoid(d1),F.sigmoid(d2),F.sigmoid(d3),F.sigmoid(d4),F.sigmoid(d5),F.sigmoid(d6),F.sigmoid(db)




