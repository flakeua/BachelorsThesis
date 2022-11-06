from email.mime import image
import os
from face_eye_finder import get_input_data
import cv2
import numpy as np

cgds_path = './datasets/columbia_gaze_data_set/'
new_dataset_path = './datasets/cgds_2/'


def transform_cgds():
    images = os.listdir(cgds_path)
    for i in images:
        params = i.split('.')[0].split('_')
        full_path = os.path.join(cgds_path, i)
        print(params)
        if len(params) != 5:
            continue
        subject = params[0]
        head_angle = params[2]
        eye_pitch = params[3]
        eye_yaw = params[4]
        image = cv2.imread(full_path)
        input_data = get_input_data(image)
        if len(input_data) == 0:
            print('No face found')
            continue
        input_data = input_data[0]
        image = input_data['image']
        new_name = '_'.join([subject, head_angle, eye_pitch, eye_yaw, str(round(input_data['p_pred_deg'].item(), 2)), str(
            round(input_data['r_pred_deg'].item(), 2)), str(round(input_data['y_pred_deg'].item(), 2))]) + '.jpg'
        image = cv2.resize(image, (900, 300))
        cv2.imwrite(os.path.join(new_dataset_path, new_name), image)


if __name__ == '__main__':
    transform_cgds()
