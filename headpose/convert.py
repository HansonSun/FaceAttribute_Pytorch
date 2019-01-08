import mobilenet
import torch,torchvision
from torch.autograd import Variable

input=Variable(torch.randn(1,3,224,224))
model =mobilenet.MobileNet( 67)
model.load_state_dict(torch.load("test_epoch_25.pkl"))
torch.onnx.export(model,input,"test.onnx")