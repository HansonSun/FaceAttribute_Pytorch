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
import datasets, hopenet, utils
import importlib


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',default='', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fd_detector=facedetect.facedetect().get_instance()
    cudnn.enabled = True

    gpu = "cuda:0"
    snapshot_path = args.snapshot

    model = importlib.import_module("hopenet").inference()

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

            img = Image.fromarray(faceimg)
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)
            yaw, pitch, roll = model(img)
            yaw.squeeze_(3)
            yaw.squeeze_(2)

            pitch.squeeze_(3)
            pitch.squeeze_(2)

            roll.squeeze_(3)
            roll.squeeze_(2)


            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            print (yaw_predicted)

            #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (det.x + det.x2) / 2, (det.y + det.y2) / 2, size = 80)
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (det.x + det.x2) / 2, tdy= (det.y + det.y2) / 2, size = 80)
            cv2.rectangle(frame, (det.x, det.y), (det.x2, det.y2), (0,255,0), 1)
        cv2.imshow("f",frame)
        cv2.waitKey(1)