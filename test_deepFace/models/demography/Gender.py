from typing import List, Union
import numpy as np

from models.facial_recognition import VGGFace
from commons import package_utils, weight_utils
from models.Demography import Demography


tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Model,Sequential
    from keras.layers import Convolution2D, Flatten, Activation
else:
    from tensorflow.keras.models import Model,Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation

WEIGHTS_URL = "https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5"

labels = ['Woman', 'Man']

class GenderClient(Demography):
    def __init__(self):
        self.model = load_model()
        self.model_name = "Gender"

    def predict(self,img:Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        imgs = self._preprocess_batch_or_single_input(img)
        predictions = self._predict_internal(imgs)
        return predictions
    
def load_model(url=WEIGHTS_URL) -> Model:
    model = VGGFace.base_model()

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1,1), name="predictions")(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    weights_file = weight_utils.download_weights_if_necessary(
        file_name = "gender_model_weights.h5", source_url = url
    )
    gender_model = weight_utils.load_model_weights(
        model = gender_model, weights_file = weights_file
    )
    return gender_model