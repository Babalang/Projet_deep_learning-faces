from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any
import numpy as np
from commons import package_utils
tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model

class FacialRecognition(ABC):
    model : Union[Model, Any]
    model_name : str
    input_shape : Tuple[int,int]
    output_shape : int

    def forward(self, img:np.ndarray) ->  Union[List[float], List[List[float]]]:
        if not isinstance(self.model, Model):
                        raise ValueError(
                "You must overwrite forward method if it is not a keras model,"
                f"but {self.model_name} not overwritten!"
            )
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        if img.ndim == 4 and img.shape[0] == 1:
            embedding = self.model(img, training=False).numpy()
        elif img.ndim == 4 and img.shape[0] > 1:
            embedding = self.model.predict_on_batch(img)
        else:
            raise ValueError("Input image must be 3 or 4 dimensional array.")
        
        assert isinstance(embedding, np.ndarray), "Model output must be a numpy array."
        if embedding.shape[0] == 1:
            return embedding[0, :].tolist()
        return embedding.tolist()