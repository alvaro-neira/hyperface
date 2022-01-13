import chainer

import cv2
import os
import numpy as np

import config
# from drawingmod import draw_pose, draw_gender, draw_detection, draw_landmark
import models

# Disable type check in chainer
os.environ["CHAINER_TYPE_CHECK"] = "0"


def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, chainer.variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v


def frame_detect(img):
    # Load config
    config.load('/Users/aneira/hyperface/config.json')

    # Define a model
    model = models.HyperFaceModel()
    model.train = False
    model.report = False
    model.backward = False

    # Initialize model
    chainer.serializers.load_npz('/Users/aneira/hyperface/model_epoch_190', model)

    xp = np

    if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        exit()
    frame = img.copy()
    frame = frame.astype(np.float32) / 255.0  # [0:1]
    frame = cv2.resize(frame, models.IMG_SIZE)
    frame = cv2.normalize(frame, None, -0.5, 0.5, cv2.NORM_MINMAX)
    frame = np.transpose(frame, (2, 0, 1))

    # Create single batch
    imgs = xp.asarray([frame])
    x = chainer.Variable(imgs)  # , volatile=True)

    y = model(x)

    # Chainer.Variable -> np.ndarray
    imgs = _cvt_variable(y['img'])
    detections = _cvt_variable(y['detection'])
    landmarks = _cvt_variable(y['landmark'])
    visibilities = _cvt_variable(y['visibility'])
    poses = _cvt_variable(y['pose'])
    genders = _cvt_variable(y['gender'])

    # Use first data in one batch
    frame = imgs[0]
    detection = detections[0]
    landmark = landmarks[0]
    visibility = visibilities[0]
    pose = poses[0]
    gender = genders[0]

    frame = np.transpose(frame, (1, 2, 0))
    frame = frame.copy()
    frame += 0.5  # [-0.5:0.5] -> [0:1]
    detection = (detection > 0.5)
    gender = (gender > 0.5)

    # Draw results
    # draw_detection(frame, detection)
    landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
    # draw_landmark(frame, landmark, visibility, landmark_color, 0.5)
    # draw_pose(frame, pose)
    # draw_gender(frame, gender)

    return 255 * frame


def test():
    lena = frame_detect(cv2.imread('/Users/aneira/hyperface/sample_images/lena_face.png'))
    cv2.imshow('result', lena / 255)
    cv2.waitKey(0)
    cv2.imwrite('/Users/aneira/hyperface/sample_images/lena_face_result3.png', lena, [cv2.IMWRITE_PNG_COMPRESSION, 0])
