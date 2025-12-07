from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np
from commons import package_utils
tf_major_version = package_utils.get_tf_major_version()
if tf_major_version == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model

class Demography(ABC):
    model: Model
    model_name: str

    @abstractmethod
    def predict(self, img:Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, np.float64]:
        pass

    def _predict_internal(self, img_batch:np.ndarray)->np.ndarray:
        if not self.model_name:
            raise NotImplementedError("no model selected")
        assert img_batch.ndim == 4, "expected 4-dimensional tensor input"
        if img_batch.shape[0] == 1:
            return self.model(img_batch,training=False).numpy()[0, :]
        return self.model.predict_on_batch(img_batch)
    
    def _preprocess_batch_or_single_input(self,img:Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        image_batch = np.array(img)
        if len(image_batch.shape) == 3:
            image_batch = np.expand_dims(image_batch, axis=0)
        return image_batch
