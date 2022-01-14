import os
from skimage import io
import cv2

from use_on_frame import frame_detect

lena = frame_detect(cv2.imread('/Users/aneira/hyperface/sample_images/2021_11_01_13_F6420.png'))
cv2.imwrite('/Users/aneira/hyperface/sample_images/lena_face_result3.png', lena, [cv2.IMWRITE_PNG_COMPRESSION, 0])
