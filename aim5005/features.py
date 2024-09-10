import numpy as np
from typing import List, Union, TypeVar

# Type aliases
ArrayLike = Union[List, np.ndarray]
NDArray = np.ndarray

def _check_is_array(x: ArrayLike) -> NDArray:
    """
    Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)
    
    assert isinstance(x, np.ndarray), "Expected the input to be a list or numpy array"
    return x

class MinMaxScaler:
    def __init__(self):
        self.minimum: NDArray = None
        self.maximum: NDArray = None
    
    def fit(self, x: ArrayLike) -> None:
        x = _check_is_array(x)
        self.minimum = np.min(x, axis=0)
        self.maximum = np.max(x, axis=0)
    
    def transform(self, x: ArrayLike) -> NDArray:
        """
        MinMax Scale the given vector
        """
        x = _check_is_array(x)
        
        # Handle division by zero for constant features
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (x - self.minimum) / (self.maximum - self.minimum)
        
        # Replace inf and -inf with nan
        result[~np.isfinite(result)] = np.nan
        
        return result
    
    def fit_transform(self, x: ArrayLike) -> NDArray:
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean: NDArray = None
        self.std: NDArray = None
    
    def fit(self, x: ArrayLike) -> None:
        x = _check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    
    def transform(self, x: ArrayLike) -> NDArray:
        """
        Standard Scale the given vector
        """
        x = _check_is_array(x)
        
        # Handle division by zero for constant features
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (x - self.mean) / self.std
        
        # Replace inf and -inf with nan
        result[~np.isfinite(result)] = np.nan
        
        return result
    
    def fit_transform(self, x: ArrayLike) -> NDArray:
        self.fit(x)
        return self.transform(x)