import chainer

import cv2
import os
import numpy as np

import config

# Disable type check in chainer
import models

os.environ["CHAINER_TYPE_CHECK"] = "0"


def _cvt_variable(v):
    # Convert from chainer variable
    if isinstance(v, chainer.variable.Variable):
        v = v.data
        if hasattr(v, 'get'):
            v = v.get()
    return v


def short_hyperface(config_path, img_path, model_path):
    # Load config
    config.load(config_path)

    # Define a model
    model = models.HyperFaceModel()
    model.train = False
    model.report = False
    model.backward = False

    # Initialize model
    chainer.serializers.load_npz(model_path, model)

    # Setup GPU
    if config.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(config.gpu).use()
        model.to_gpu()
        xp = chainer.cuda.cupy
    else:
        xp = np

    # Load image file
    img = cv2.imread(img_path)
    if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        exit()
    img = img.astype(np.float32) / 255.0  # [0:1]
    img = cv2.resize(img, models.IMG_SIZE)
    img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
    img = np.transpose(img, (2, 0, 1))

    # Create single batch
    imgs = xp.asarray([img])
    x = chainer.Variable(imgs)  # , volatile=True)

    # Forward
    y = model(x)

    # Chainer.Variable -> np.ndarray
    detection = _cvt_variable(y['detection'])
    genders = _cvt_variable(y['gender'])

    gender = genders[0]

    if gender > 0.5:
        return f"Female, detection={detection}"
    else:
        return f"Male, detection={detection}"


print(short_hyperface('/Users/aneira/noticias/hyperface/config.json',
                      '/Users/aneira/noticias/hyperface/sample_images/lena_face.png',
                      '/Users/aneira/hyperface/model_epoch_190'))
