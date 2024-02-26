from face_detection import RetinaFace
from sixdrepnet.model import SixDRepNet
import os
import numpy as np
import cv2
from math import cos, sin

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from sixdrepnet import utils
import matplotlib
from PIL import Image
import time

from networks import *

import torch
import torch_directml

matplotlib.use('TkAgg')


transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dml = torch_directml.device()
cudnn.enabled = True
cam = 1
snapshot_path = "poc/models/6DRepNet_300W_LP_AFLW2000.pth"
device = dml
model = SixDRepNet(backbone_name='RepVGG-B1g2',
                   backbone_file='',
                   deploy=True,
                   pretrained=False)

print('Loading data.')

detector = RetinaFace(gpu_id=device)

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


def get_input_data(image, offset_coeff=1) -> dict:
    try:
        coeff = 1280 / image.shape[1]
        resized_image = cv2.resize(image, (1280, int(image.shape[0]*coeff)))
        with torch.no_grad():
            faces = detector(resized_image)
            result = []
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

                x_3 = int(landmarks[0][0])
                y_3 = int(landmarks[0][1])
                x_4 = int(landmarks[1][0])
                y_4 = int(landmarks[1][1])

                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max += int(0.2*bbox_height)
                y_max += int(0.2*bbox_width)

                img = resized_image[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)
                img = img.convert('RGB')
                img = transformations(img)

                img = torch.Tensor(img[None, :]).to(device)

                c = cv2.waitKey(1)
                if c == 27:
                    break

                R_pred = model(img)

                euler = utils.compute_euler_angles_from_rotation_matrices(
                    R_pred)*180/np.pi

                curr = {'p_pred_deg': euler[:, 0].cpu(),
                        'y_pred_deg': euler[:, 1].cpu(),
                        'r_pred_deg': euler[:, 2].cpu()
                        }

                offset = abs(((x_3 - x_min2)/2 + (x_max2-x_4)/2)/2)
                x_offset = int(offset*1.2*offset_coeff)
                y_offset = int(offset*0.8*offset_coeff)

                y_3_min = int((y_3 - y_offset) / coeff)
                y_3_max = int((y_3 + y_offset) / coeff)
                x_3_min = int((x_3 - x_offset) / coeff)
                x_3_max = int((x_3 + x_offset) / coeff)

                y_4_min = int((y_4 - y_offset) / coeff)
                y_4_max = int((y_4 + y_offset) / coeff)
                x_4_min = int((x_4 - x_offset) / coeff)
                x_4_max = int((x_4 + x_offset) / coeff)

                right_eye = image[y_3_min:y_3_max, x_3_min: x_3_max]
                left_eye = image[y_4_min:y_4_max, x_4_min: x_4_max]
                left_eye = cv2.resize(
                    left_eye, (right_eye.shape[1], right_eye.shape[0]))
                curr['image'] = cv2.hconcat([right_eye, left_eye])
                curr['box'] = list(map(lambda x: x/coeff, box))
                curr['landmarks'] = list(
                    map(lambda y: list(map(lambda x: x/coeff, y)), landmarks))
                result.append(curr)
    except Exception as e:
        print(e.args)
        return None
    return result


def draw_eye_axis(img, yaw, pitch, roll, tdx, tdy, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    x = size * (sin(yaw)) + tdx
    y = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x), int(y)), (255, 255, 0), 3)

    return img


if __name__ == '__main__':

    transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((70, 210)),
                                     transforms.ToTensor()])
    cap = cv2.VideoCapture(cam)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    net = SixthEyeNet()
    EYE_MODEL_PATH = './poc/models/sixth_eye_net_combined.pth'
    bw = False
    net.load_state_dict(torch.load(EYE_MODEL_PATH))
    net.to(device)
    # torch.save(net.state_dict(), './code/poc/models/thrird_eye_net_cuda.pth')
    with torch.no_grad():
        n = 0
        while True:
            coeff = 1
            _, frame = cap.read()
            # images = os.listdir('./datasets/me_test/')
            # coeff = 1
            # frame = cv2.imread(
            #     f'./datasets/me_test/{images[n]}')
            input_data = get_input_data(frame)
            if input_data is None:
                continue
            if len(input_data) == 0:
                continue

            for face in input_data:
                box = face['box']
                landmarks = face['landmarks']
                # Print the location of each face in this image
                x_min = int(box[0])
                y_min = int(box[1])
                x_max = int(box[2])
                y_max = int(box[3])

                x_min2 = int(box[0])
                y_min2 = int(box[1])
                x_max2 = int(box[2])
                y_max2 = int(box[3])

                x_3 = int(landmarks[0][0])
                y_3 = int(landmarks[0][1])
                x_4 = int(landmarks[1][0])
                y_4 = int(landmarks[1][1])

                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)

                x_min = max(0, x_min-int(0.2*bbox_height))
                y_min = max(0, y_min-int(0.2*bbox_width))
                x_max += int(0.2*bbox_height)
                y_max += int(0.2*bbox_width)

                hp = face['p_pred_deg']
                hy = face['y_pred_deg']
                hr = face['r_pred_deg']

                image = face['image']
                image = cv2.resize(image, (210, 70),
                                   interpolation=cv2.INTER_CUBIC)
                if bw:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = transforms(image)
                head_pos = torch.unsqueeze(torch.tensor(
                    [float(hp), float(hr), float(hy)], dtype=torch.float32), dim=0).to(device)
                image = torch.unsqueeze(image, dim=0).to(device)
                res = net((image, head_pos))
                res = res.tolist()[0]
                pitch = res[0]
                yaw = -res[1]
                new_yaw = head_pos.cpu().data.numpy()[0][2]
                if abs(new_yaw) > 30:
                    yaw = new_yaw
                print(pitch, yaw)

                x_3 = int(landmarks[0][0])
                y_3 = int(landmarks[0][1])
                x_4 = int(landmarks[1][0])
                y_4 = int(landmarks[1][1])

                # utils.draw_axis(frame, yaw, pitch, 0, x_3,
                #                 y_3, size=60, thickness=2)
                # utils.draw_axis(frame, yaw, pitch, 0, x_4,
                #                 y_4, size=60, thickness=2)

                utils.draw_axis(frame, yaw, pitch, hr,
                                x_min+int(.5*(x_max-x_min)), y_min+int(.5*(y_max-y_min)), size=130*coeff)

                # utils.draw_axis(frame, hy, hp, hr,
                #                 x_min+int(.5*(x_max-x_min)) + 100, y_min+int(.5*(y_max-y_min)), size=130)

            cv2.imshow("Demo", frame)
            # cv2.imwrite(f'./images/face-{n}.jpg', frame)
            cv2.waitKey(1)
            n += 1
