import time
from typing import List, Tuple, Optional, Union, Generator, Dict, Any
import numpy as np
import math


def l2_normalize(
    x: Union[np.ndarray, list], axis: Union[int, None] = None, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
        axis (int): axis along which to normalize
    Returns:
        np.ndarray: l2 normalized vector
    """
    # Convert inputs to numpy arrays if necessary
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + epsilon)