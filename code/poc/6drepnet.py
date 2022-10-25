from pprint import pprint
from face_detection import RetinaFace
from SixDRepNet.model import SixDRepNet
import math
import re
from matplotlib import pyplot as plt
import sys
import os
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.function_base import _quantile_unchecked

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from SixDRepNet import utils
import matplotlib
from PIL import Image
import time
matplotlib.use('TkAgg')


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    cudnn.enabled = True
    gpu = 0
    cam = 0
    snapshot_path = "code/poc/models/6DRepNet_300W_LP_AFLW2000.pth"
    device = "mps"
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False, gpu_id=gpu)

    print('Loading data.')

    detector = RetinaFace(gpu_id=gpu)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()

            faces = detector(frame)

            for box, landmarks, score in faces:

                # Print the location of each face in this image
                if score < .95:
                    continue
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])

                x_min2 = int(box[0])
                y_min2 = int(box[1])
                x_max2 = int(box[2])
                y_max2 = int(box[3])

                x_min3 = int(landmarks[0][0])
                y_min3 = int(landmarks[0][1])
                x_max3 = int(landmarks[1][0])
                y_max3 = int(landmarks[1][1])

                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max += int(0.2*bbox_height)
                y_max += int(0.2*bbox_width)

                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).to(device)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                start = time.time()
                R_pred = model(img)
                end = time.time()
                print('Head pose estimation: %2f ms' % ((end - start)*1000.))

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred, use_gpu=True)*180/np.pi
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                utils.draw_axis(frame, y_pred_deg, p_pred_deg, r_pred_deg,
                                x_min+int(.5*(x_max-x_min)), y_min+int(.5*(y_max-y_min)), size=130)

                cv2.rectangle(frame, (x_min2, y_min2),
                              (x_max2, y_max2), (170, 170, 0), 5)
                offset = ((x_min3 - x_min2)/2 + (x_max2-x_max3)/2)/2
                x_offset = int(offset*1.2)
                y_offset = int(offset*0.8)
                cv2.rectangle(frame, (x_min3 - x_offset, y_min3 - y_offset),
                              (x_max3 + x_offset, y_max3 + y_offset), (170, 0, 170), 5)

            cv2.imshow("Demo", frame)
            cv2.waitKey(5)
