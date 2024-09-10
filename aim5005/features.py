import numpy as np
from typing import List, Tuple

def _check_is_array(x: np.ndarray) -> np.ndarray:
    """
    Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    assert isinstance(x, np.ndarray), "Expected the input to be a list"
    return x

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
    
    def fit(self, x: np.ndarray) -> None:   
        x = _check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = _check_is_array(x)
        
        # Fixed: Added parentheses to ensure correct order of operations
        return (x - self.minimum) / (self.maximum - self.minimum)
    
    def fit_transform(self, x: list) -> np.ndarray:
        x = _check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, x: np.ndarray) -> None:
        x = _check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    
    def transform(self, x: np.ndarray) -> list:
        """
        Standard Scale the given vector
        """
        x = _check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: list) -> np.ndarray:
        x = _check_is_array(x)
        self.fit(x)
        return self.transform(x)