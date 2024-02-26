import os
from face_eye_finder import get_input_data
import cv2


def transform_cgds(cgds_path, new_dataset_path):
    images = os.listdir(cgds_path)
    n = 0
    for i in images:
        params = i.split('.')[0].split('_')
        full_path = os.path.join(cgds_path, i)
        # print(params)
        if len(params) < 5:
            continue
        subject = params[0]
        head_angle = params[2]
        eye_pitch = params[3]
        eye_yaw = params[4]
        image = cv2.imread(full_path)
        input_data = get_input_data(image, 0.7)
        if len(input_data) == 0:
            print('No face found')
            continue
        input_data = input_data[0]
        image = input_data['image']
        new_name = '_'.join([subject, head_angle, eye_pitch, eye_yaw, str(round(input_data['p_pred_deg'].item(), 2)), str(
            round(input_data['r_pred_deg'].item(), 2)), str(round(input_data['y_pred_deg'].item(), 2)), str(n)]) + '.jpg'
        image = cv2.resize(image, (210, 70), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(new_dataset_path, new_name), image)
        n += 1


def mirror_cgds(cgds_path, new_dataset_path):
    images = os.listdir(cgds_path)
    for i in images:
        params = i.split('.')[0].split('_')
        if len(params) != 5:
            continue
        params[4] = f'{str(-int(params[4][:-1]))}H'
        params.append('m')
        full_path = os.path.join(cgds_path, i)
        image = cv2.imread(full_path)
        cv2.imwrite(os.path.join(new_dataset_path, i), image)
        image = cv2.flip(image, 1)
        new_name = ('_'.join(params) + '.jpg')
        cv2.imwrite(os.path.join(new_dataset_path, new_name), image)
        print(i)


def rename_cgds(cgds_path, new_dataset_path):
    images = os.listdir(cgds_path)
    for i in images:
        params = i.split('.')[0].split('_')
        if len(params) != 5:
            continue
        params[4] = f'{str(int(params[4][:-1]) - int(params[2][:-1]))}H'
        full_path = os.path.join(cgds_path, i)
        image = cv2.imread(full_path)
        new_name = ('_'.join(params) + '.jpg')
        cv2.imwrite(os.path.join(new_dataset_path, new_name), image)
        print(i)


if __name__ == '__main__':
    # rename_cgds('./datasets/columbia_gaze_data_set', './datasets/cgds_5')
    mirror_cgds('./datasets/columbia_gaze_data_set',
                './datasets/cgds_mirrored')
    transform_cgds('./datasets/cgds_mirrored/', './datasets/cgds_m/')
