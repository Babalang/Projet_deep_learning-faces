from typing import Tuple
import cv2
import numpy as np
from commons import package_utils
tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.preprocessing import image
else:
    from tensorflow.keras.preprocessing import image

def normalizze_input(img:np.ndarray, normalization: str = "base")->np.ndarray:
    if normalization == "base":
        return img
    img*=255
    if normalization == "raw":
        pass
    elif normalization == "Facenet":
        img /= 127.5
        img -= 1.0
    elif normalization == "VGGFace":
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
    elif normalization == "VGGFace2":
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912
    elif normalization == "ArcFace":
        img /= 127.5
        img -= 1.0
    else:
        raise ValueError(f"Normalization method '{normalization}' is not recognized.")
    return img

def resize(img:np.ndarray, target_size:Tuple[int, int])->np.ndarray:
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)
    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]

    img = np.pad(
        img,
        (
            (diff_0 // 2, diff_0 - diff_0 // 2),
            (diff_1 // 2, diff_1 - diff_1 // 2),
            (0, 0),
        ),
        'constant',
    )
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    if img.max()>1:
        img = (img.astype(np.float32)/255.0).astype(np.float32)
    return img