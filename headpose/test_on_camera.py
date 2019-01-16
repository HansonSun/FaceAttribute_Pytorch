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
import  utils
import importlib


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',default='', type=str)
    args = parser.parse_args()
    return args


def img_preprocess(img):
    processimg=cv2.resize(img,(112,112))
    processimg=processimg.astype(np.float32)
    processimg=np.transpose(processimg,(2,0,1))
    processimg=np.expand_dims(processimg,0)
    processimg=processimg/255.0
    processimg=(processimg-0.4)/0.2
    return processimg


if __name__ == '__main__':
    args = parse_args()
    fd_detector=facedetect.facedetect().get_instance()
    cudnn.enabled = True

    gpu = "cuda:0"
    snapshot_path = args.snapshot

    model = importlib.import_module("resnet").inference()

    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Resize(112),
    transforms.CenterCrop(112), transforms.ToTensor(),
    transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])])

    model.cuda(gpu)
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    idx_tensor = [idx for idx in range(67)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    camera = cv2.VideoCapture(0)

    while True:
        ret,frame = camera.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        dets = fd_detector.findfaces(cv2_frame)

        for idx, det in enumerate(dets):

            faceimg = det.get_roi(cv2_frame)
            faceimg=img_preprocess(faceimg)

            img = torch.from_numpy(faceimg).cuda("cuda:0")
            yaw, pitch, roll = model(img)
            yaw.squeeze_(3)
            yaw.squeeze_(2)

            pitch.squeeze_(3)
            pitch.squeeze_(2)

            roll.squeeze_(3)
            roll.squeeze_(2)


            yaw_predicted = F.softmax(yaw,dim=1)
            pitch_predicted = F.softmax(pitch,dim=1)
            roll_predicted = F.softmax(roll,dim=1)

            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
            #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (det.x + det.x2) / 2, (det.y + det.y2) / 2, size = 80)
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (det.x + det.x2) / 2, tdy= (det.y + det.y2) / 2, size = 80)
            cv2.rectangle(frame, (det.x, det.y), (det.x2, det.y2), (0,255,0), 1)
        cv2.imshow("f",frame)
        cv2.waitKey(1)