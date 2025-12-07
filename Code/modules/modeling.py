from __future__ import annotations

from typing import List, Generator, Dict, Any, TYPE_CHECKING, Final,TypedDict


from models.demography import Gender, Emotion
from models.face_detection import OpenCv

if TYPE_CHECKING:
    from models.Demography import Demography
    from test_deepFace.models.Detector import Detector

    class AvailableModels(TypedDict):
        facial_attribute: dict[str, type[Demography]]
        face_detector: dict[str, type[Detector]]
        
AVAILABLE_MODELS: Final[AvailableModels] = {
    "facial_attribute": {
        "Gender": Gender.GenderClient,
        "Emotion": Emotion.EmotionClient,
    },
    "face_detector": {
        "opencv": OpenCv.OpenCvClient,
    },
}

def build_model(task:str, model_name:str)-> Any:
    """
    Build a model for a given task and model name
    Args:
        task (str): task type (e.g., 'facial_attribute', 'face_detector')
        model_name (str): name of the model to build
    Returns:
        model (Any): instantiated model
    """
    global cached_models
    if task not in AVAILABLE_MODELS.keys():
        raise ValueError(f"Task '{task}' is not supported. Supported tasks are: {list(AVAILABLE_MODELS.keys())}")
    if "cached_models" not in globals():
        cached_models = {current_task: {} for current_task in AVAILABLE_MODELS.keys()}

    if cached_models[task].get(model_name) is None:
        model = AVAILABLE_MODELS[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else :
            raise ValueError(f"Model '{model_name}' is not available for task '{task}'. Available models are: {list(AVAILABLE_MODELS[task].keys())}")
    return cached_models[task][model_name]