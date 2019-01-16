import sys, os, argparse
sys.path.append("/home/hanson/facetools/lib")
sys.path.append("model")

import facedetect
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import utils
import matplotlib.pyplot as plt
import importlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',default='', type=str)
    args = parser.parse_args()
    return args

class EularAngle():
    def __init__(self,modelfile):
        self.fd_detector=facedetect.facedetect().get_instance()
        cudnn.enabled = True
        self.gpu = "cuda:0"
        self.model = importlib.import_module("resnet").inference()

        saved_state_dict = torch.load(modelfile)
        self.model.load_state_dict(saved_state_dict)
        self.transformations = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])])
        self.model.cuda(self.gpu)
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).


    def img_preprocess(self,img):
        processimg=cv2.resize(img,(112,112))
        processimg=processimg.astype(np.float32)
        processimg=np.transpose(processimg,(2,0,1))
        processimg=np.expand_dims(processimg,0)
        processimg=processimg/255.0
        processimg=(processimg-0.4)/0.2
        return processimg

    def getangle(self,imagepath):
        idx_tensor = [idx for idx in xrange(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu)

        frame=cv2.imread(imagepath)
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        det = self.fd_detector.findbiggestface(cv2_frame)
        faceimg = det.get_roi(cv2_frame)

        faceimg=self.img_preprocess(faceimg)

        img = torch.from_numpy(faceimg).cuda(self.gpu)

        yaw, pitch, roll = self.model(img)

        yaw_predicted   = F.softmax(yaw,dim=1)
        pitch_predicted = F.softmax(pitch,dim=1)
        roll_predicted  = F.softmax(roll,dim=1)
        # Get continuous predictions in degrees.
        yaw_predicted = float((torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99 ).cpu().numpy() )
        pitch_predicted = float( (torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99).cpu().numpy() )
        roll_predicted = float( (torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99).cpu().numpy() )
        return (roll_predicted ,pitch_predicted ,yaw_predicted)


    def getangle_nodetect(self,imagepath):
        idx_tensor = [idx for idx in xrange(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu)

        frame=cv2.imread(imagepath)
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        faceimg=self.img_preprocess(cv2_frame)

        img = torch.from_numpy(faceimg).cuda(self.gpu)

        yaw, pitch, roll = self.model(img)

        yaw_predicted   = F.softmax(yaw,dim=1)
        pitch_predicted = F.softmax(pitch,dim=1)
        roll_predicted  = F.softmax(roll,dim=1)

        yaw_predicted = float((torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99 ).cpu().numpy() )
        pitch_predicted = float( (torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99).cpu().numpy() )
        roll_predicted = float( (torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99).cpu().numpy() )
        return (roll_predicted ,pitch_predicted ,yaw_predicted)


if __name__=="__main__":
    args=parse_args()


    result_roll=[]
    result_pitch=[]
    result_yaw=[]
    demo=EularAngle(args.snapshot)
    for root,_,filenames in os.walk("/home/hanson/dataset/CelebA/Img/img_celeba.7z/img_celeba"):
        for filename in filenames :
            imgpath=os.path.join(root,filename)
            roll,pitch,yaw= (demo.getangle(imgpath) )
            result_roll.append(roll)
            result_pitch.append(pitch)
            result_yaw.append(yaw)

    fig,ax= plt.subplots( figsize=(17, 8))
    ax.hist(result_roll, list(np.arange(-90,90,1) ), density=True, histtype='bar', color="r",rwidth=0.8)
    ax.set_title('roll')
    fig.tight_layout()


    fig1,ax1= plt.subplots( figsize=(17, 8))
    ax1.hist(result_pitch, range(-90,90,1), density=True, histtype='bar',color="b", rwidth=0.8)
    ax1.set_title('pitch')
    fig1.tight_layout()


    fig2,ax2= plt.subplots( figsize=(17, 8))
    ax2.hist(result_yaw, range(-90,90,1), density=True, histtype='bar',color="g", rwidth=0.8)
    ax2.set_title('yaw')
    fig2.tight_layout()
    plt.show()