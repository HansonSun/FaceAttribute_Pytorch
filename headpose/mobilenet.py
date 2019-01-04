import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

class MobileNet(nn.Module):
    def __init__(self,num_bins):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            #nn.AvgPool2d(7),
            nn.MaxPool2d(2)
        )
        self.fc_yaw   = nn.Conv2d(1024, num_bins,kernel_size=1)
        self.fc_pitch = nn.Conv2d(1024, num_bins,kernel_size=1)
        self.fc_roll  = nn.Conv2d(1024, num_bins,kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        pre_yaw   = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll  = self.fc_roll(x)

        pre_yaw.squeeze_(3)
        pre_yaw.squeeze_(2)

        pre_pitch.squeeze_(3)
        pre_pitch.squeeze_(2)

        pre_roll.squeeze_(3)
        pre_roll.squeeze_(2)

        return pre_yaw, pre_pitch, pre_roll  


if __name__ == '__main__':
    net = MobileNet(67)
    x = torch.randn(1, 3, 60, 60)
    yaw,pitch,roll = net(Variable(x))
    print (yaw.shape)
    torch.onnx.export(net,x,"test.onnx")
