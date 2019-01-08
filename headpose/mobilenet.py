import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

class MobileNet(nn.Module):
    def __init__(self,num_bins,depth_multiplier=1.0):
        super(MobileNet, self).__init__()
        depth = lambda d: max(int(d * depth_multiplier), 8)

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
            conv_bn(  3, depth(32), 2), 
            conv_dw( depth(32)  , depth(64)  , 1),
            conv_dw( depth(64)  , depth(128) , 2),
            conv_dw( depth(128) , depth(128) , 1),
            conv_dw( depth(128) , depth(256) , 2),
            conv_dw( depth(256) , depth(256) , 1),
            conv_dw( depth(256) , depth(512) , 2),
            conv_dw( depth(512) , depth(512) , 1),
            conv_dw( depth(512) , depth(512) , 1),
            conv_dw( depth(512) , depth(512) , 1),
            conv_dw( depth(512) , depth(512) , 1),
            conv_dw( depth(512) , depth(512) , 1),
            conv_dw( depth(512) , depth(1024), 2),
            conv_dw( depth(1024), depth(1024), 1),    
        )
        self.fc_yaw   = nn.Conv2d(depth(1024), num_bins,kernel_size=1)
        self.fc_pitch = nn.Conv2d(depth(1024), num_bins,kernel_size=1)
        self.fc_roll  = nn.Conv2d(depth(1024), num_bins,kernel_size=1)
        self.global_pool=nn.MaxPool2d(7)


    def forward(self, x):
        x = self.model(x)
        #print(x.shape)
        x=self.global_pool(x)
        #print (x.shape)


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


    net.eval()
    x = torch.randn(1, 3, 112, 112)

    #torch.save(net.state_dict(),'test.pkl')
    yaw,pitch,roll = net(Variable(x))
    #print (yaw.shape)
    #torch.onnx.export(net,x,"test.onnx")


