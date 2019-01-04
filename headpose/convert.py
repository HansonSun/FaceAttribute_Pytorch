import hopenet2
import hopenet
import torch,torchvision
from torch.autograd import Variable

input=Variable(torch.randn(1,3,224,224))
model = hopenet2.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 67)
model.load_state_dict(torch.load("./test_epoch_5.pkl"))
torch.onnx.export(model,input,"test.onnx")