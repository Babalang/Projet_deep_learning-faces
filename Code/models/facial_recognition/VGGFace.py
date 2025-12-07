from typing import List

import numpy as np 

from commons import package_utils, weight_utils
from modules import verification
from models.FacialRecognition import FacialRecognition

tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"
)

class VggFaceClient(FacialRecognition):
    def __init__(self):
        self.model = load_model()
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096

        def forward(self,img:np.ndarray)-> List[float]:
            embedding = super().forward(img)
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = verification.l2_normalize(embedding, axis=1)
            else:
                embedding = verification.l2_normalize(embedding)
            return embedding.tolist()
        
def base_model() -> Sequential:
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))

    return model

def load_model(url=WEIGHTS_URL) -> Model:
    model = base_model()
    weights_file = weight_utils.download_weights_if_necessary(
        file_name="vgg_face_weights.h5", source_url=url
    )
    model = weight_utils.load_model_weights(model=model, weights_file=weights_file)
    base_model_output = Flatten()(model.layers[-5].output)
    vgg_face_descriptior = Model(inputs = model.layers[0].input, outputs = base_model_output)
    return vgg_face_descriptior