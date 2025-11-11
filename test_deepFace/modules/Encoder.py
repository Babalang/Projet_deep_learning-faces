# Fichier python pour l'encodage des visages, basée sur l'exemple de DeepFace

# common dependencies
import os
import warnings
import cv2
import logging
from typing import Any, Dict, IO, List, Union, Optional, Sequence

from tqdm import tqdm


# this has to be set before importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# 3rd party dependencies
import numpy as np
import pandas as pd
import tensorflow as tf

from modules.detection import extract_faces
import modules.modeling as modeling
import modules.preprocessing as preprocessing
from models.demography import Gender,Emotion

# Fonction pour anaalyser un visage etretourner les caractéristiques [Genre, Age, Emotion]
def analyze_face(
    img_path: Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]],
    actions: Union[tuple, list] = ("emotion", "gender"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    # Dans le cas d'une liste d'images, traiter chaque image individuellement
    if (isinstance(img_path, np.ndarray) and img_path.ndim == 4 and img_path.shape[0]>1) or (isinstance(img_path, list)):
        batch_resp_obj = []
        for single_img in img_path:
            resp_obj = analyze_face(
                img_path=single_img,
                actions=actions,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
                align=align,
                expand_percentage=expand_percentage,
                silent=silent,
                anti_spoofing=anti_spoofing,
            )
            batch_resp_obj.append(resp_obj)
        return batch_resp_obj
    
    # Dans le cas d'une seule image
    if(isinstance(actions, str)):
        actions = (actions,)

    # test si les actions sont valides
    if not hasattr(actions, "__getitem__") or not actions :
        raise ValueError("Actions parameter must be a list or tuple of actions.")
    
    actions = list(actions)

    for action in actions:
        if(action not in ("emotion", "gender")):
            raise ValueError(f"Action '{action}' is not supported. Supported actions are 'emotion', 'age', 'gender'.")
    
    resp_objects = []
    
    img_objs = extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        grayscale=False,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )
    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoofing detected! The image is not a real face.")
        
        img_content = img_obj["face"]
        img_region = img_obj["facial_area"]
        img_confidence = img_obj["confidence"]
        if img_content.shape[0] == 0 or img_content.shape[1] == 0:
            continue
            
        img_content = img_content[:,:,::-1]

        img_content = preprocessing.resize(img=img_content, target_size=(224, 224))

        obj = {}
        pbar = tqdm(range(0, len(actions)), desc="Finding actions", disable=silent if len(actions)>1 else True,)
        for index in pbar:
            action = actions[index]
            pbar.set_description(f"Finding {action}")

            if action == "emotion":
                emotion_predictions = modeling.build_model(task = "facial_attribute", model_name = "Emotion").predict(img_content)
                sum_of_predictions = emotion_predictions.sum()

                obj["emotion"] = {}
                for i,emotion_label in enumerate(Emotion.labels):
                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                    obj["emotion"][emotion_label] = emotion_prediction
                obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

            elif action == "gender":
                gender_predictions = modeling.build_model(task = "facial_attribute", model_name = "Gender").predict(img_content)
                obj["gender"] = {}
                for i,gender_label in enumerate(Gender.labels):
                    gender_prediction = 100 * gender_predictions[i]
                    obj["gender"][gender_label] = gender_prediction
                obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

            obj["face_confidence"] = img_confidence
            obj["facial_area"] = img_region
        resp_objects.append(obj)
    return resp_objects
    