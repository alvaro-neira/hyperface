import os
from skimage import io
import cv2

from use_on_frame import frame_detect

for filename in os.listdir("/Users/aneira/hyperface/faces"):
    if not filename.endswith(".png"):
        continue
    img_path = os.path.join("/Users/aneira/hyperface/faces", filename)
    image_swapped = io.imread(f"{img_path}")
    image_2 = cv2.cvtColor(image_swapped, cv2.COLOR_BGR2RGB)
    image3 = frame_detect(image_2)
    # cv2_imshow(image3)
    cv2.imwrite(f'/Users/aneira/hyperface/results/result_{filename}', image3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(filename)
    print('\n\n')
