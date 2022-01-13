import chainer

import cv2
import os
import numpy as np

import config
from drawing import draw_pose, draw_gender, draw_detection, draw_landmark
import log_initializer
import models

# logging
from logging import getLogger, DEBUG

log_initializer.setFmt()
log_initializer.setRootLevel(DEBUG)
logger = getLogger(__name__)

# Disable type check in chainer
os.environ["CHAINER_TYPE_CHECK"] = "0"


def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, chainer.variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v


def frame_detect(frame):
    # Load config
    config.load('config.json')

    # Define a model
    logger.info('Define a HyperFace model')
    model = models.HyperFaceModel()
    model.train = False
    model.report = False
    model.backward = False

    # Initialize model
    chainer.serializers.load_npz('/content/drive/MyDrive/ai/model_epoch_190', model)

    xp = np

    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        logger.error('Failed to load')
        exit()
    frame = frame.astype(np.float32) / 255.0  # [0:1]
    frame = cv2.resize(frame, models.IMG_SIZE)
    frame = cv2.normalize(frame, None, -0.5, 0.5, cv2.NORM_MINMAX)
    frame = np.transpose(frame, (2, 0, 1))

    # Create single batch
    imgs = xp.asarray([frame])
    x = chainer.Variable(imgs)  # , volatile=True)

    # Forward
    logger.info('Forward the network')
    y = model(x)

    # Chainer.Variable -> np.ndarray
    imgs = _cvt_variable(y['img'])
    detections = _cvt_variable(y['detection'])
    landmarks = _cvt_variable(y['landmark'])
    visibilitys = _cvt_variable(y['visibility'])
    poses = _cvt_variable(y['pose'])
    genders = _cvt_variable(y['gender'])

    # Use first data in one batch
    frame = imgs[0]
    detection = detections[0]
    landmark = landmarks[0]
    visibility = visibilitys[0]
    pose = poses[0]
    gender = genders[0]

    frame = np.transpose(frame, (1, 2, 0))
    frame = frame.copy()
    frame += 0.5  # [-0.5:0.5] -> [0:1]
    detection = (detection > 0.5)
    gender = (gender > 0.5)

    # Draw results
    draw_detection(frame, detection)
    landmark_color = (0, 1, 0) if detection == 1 else (0, 0, 1)
    draw_landmark(frame, landmark, visibility, landmark_color, 0.5)
    draw_pose(frame, pose)
    draw_gender(frame, gender)

    # Show image
    logger.info('Show the result image')
    cv2.imshow('result', frame)
    cv2.waitKey(0)


frame_detect(cv2.imread('sample_images/lena_face.png'))
