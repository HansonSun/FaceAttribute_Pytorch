import sys, os, argparse
sys.path.append("/home/hanson/facetools/lib")
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
import hopenet2


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

    model = hopenet2.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 67)

    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
            x_min = det.left
            y_min = det.top
            x_max = det.right
            y_max = det.bottom
            conf = det.confidence

            if conf > 0.5:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)

                img = cv2_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)
                print (img.shape)
                yaw, pitch, roll = model(img)


                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                #print (yaw)
                #print (yaw_predicted)
                #print(yaw_predicted.sum())
                #print(torch.sum(yaw_predicted.data[0] * idx_tensor))

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
        cv2.imshow("f",frame)
        cv2.waitKey(1)