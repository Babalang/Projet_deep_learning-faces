from typing import List, Union
import numpy as np
import cv2

from models.Demography import Demography
from commons import package_utils, weight_utils

tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Model,Sequential
    from keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, Dropout, Dense
else:
    from tensorflow.keras.models import Model,Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Activation, MaxPooling2D, AveragePooling2D, Dropout, Dense

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"

class EmotionClient(Demography):
    def __init__(self):
        self.model = load_model()
        self.model_name = "Emotion"

    def _preprocess_image(self, img:np.ndarray) -> np.ndarray:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        return img_gray

    def predict(self,img:Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        imgs = self._preprocess_batch_or_single_input(img)
        processed_imgs = np.expand_dims(np.array([self._preprocess_image(img) for img in imgs]), axis=-1)
        predictions = self._predict_internal(processed_imgs)
        return predictions
    
def load_model(
    url=WEIGHTS_URL,
) -> Sequential:
    """
    Consruct emotion model, download and load weights
    """

    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    # ----------------------------

    weight_file = weight_utils.download_weights_if_necessary(
        file_name="facial_expression_model_weights.h5", source_url=url
    )

    model = weight_utils.load_model_weights(model=model, weights_file=weight_file)

    return model
