from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> List["FacialAreaRegion"]:
        pass

@dataclass
class FacialAreaRegion:
    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    confidence: Optional[float] = None
    nose: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None

@dataclass
class DetectedFace:
    img:np.array
    facial_area: FacialAreaRegion
    confidence: float