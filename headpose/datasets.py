import os
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter

import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, transform,data_dir="", filename_path="./label/300W_LP_PATH.txt", img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw   = pose[1] * 180 / np.pi
        roll  = pose[2] * 180 / np.pi

        #blur img
        #ds = 1 + np.random.randint(0,4) 
        #original_size = img.size
        #img = img.resize((int(img.size[0] / ds), int(img.size[1] / ds) ), resample=Image.NEAREST)
        #img = img.resize((int(original_size[0]), int(original_size[1]) ), resample=Image.NEAREST)


        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        #rotate img
        #rnd_angle = np.random.uniform(-40,40)
        #roll=roll-rnd_angle
        #img=img.rotate(rnd_angle)

        # Bin values
        bins = np.array(range(-99, 99, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins)

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length



if __name__ == "__main__":
    testdataset = Pose_300W_LP(None)

    for img, labels, cont_labels, imgpath in testdataset:
        nimg=np.array(img)
        print ("yaw",cont_labels[0],"pitch",cont_labels[1],"row",cont_labels[2])
        cv2.imshow("e",nimg)
        cv2.waitKey(0) 