import numpy as np
from typing import List, Union, Dict

# Type aliases
ArrayLike = Union[List, np.ndarray]
NDArray = np.ndarray

def _check_is_array(x: ArrayLike) -> NDArray:
    """
    Convert x to a np.ndarray if it's not already one. Raise an error if it can't be cast.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)
    assert isinstance(x, np.ndarray), "Expected the input to be a list or numpy array"
    return x

class MinMaxScaler:
    def __init__(self):
        self.minimum: NDArray = None
        self.maximum: NDArray = None

    def fit(self, x: ArrayLike) -> 'MinMaxScaler':
        x = _check_is_array(x)
        self.minimum = np.min(x, axis=0)
        self.maximum = np.max(x, axis=0)
        return self

    def transform(self, x: ArrayLike) -> NDArray:
        """
        MinMax Scale the given vector
        """
        x = _check_is_array(x)
        if self.minimum is None or self.maximum is None:
            raise ValueError("Scaler is not fitted. Call 'fit' before using this method.")

        # Handle division by zero for constant features
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (x - self.minimum) / (self.maximum - self.minimum)

        # Replace inf and -inf with nan
        result[~np.isfinite(result)] = np.nan

        return result

    def fit_transform(self, x: ArrayLike) -> NDArray:
        return self.fit(x).transform(x)

class StandardScaler:
    def __init__(self):
        self.mean: NDArray = None
        self.std: NDArray = None

    def fit(self, x: ArrayLike) -> 'StandardScaler':
        x = _check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return self

    def transform(self, x: ArrayLike) -> NDArray:
        """
        Standard Scale the given vector
        """
        x = _check_is_array(x)
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted. Call 'fit' before using this method.")

        # Handle division by zero for constant features
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (x - self.mean) / self.std

        # Replace inf and -inf with nan
        result[~np.isfinite(result)] = np.nan

        return result

    def fit_transform(self, x: ArrayLike) -> NDArray:
        return self.fit(x).transform(x)

class LabelEncoder:
    def __init__(self):
        self.classes_: NDArray = None
        self._label_dict: Dict[Union[str, int], int] = {}

    def fit(self, y: ArrayLike) -> 'LabelEncoder':
        """
        Fit label encoder

        Args:
            y : array-like of shape(n_samples,)
                Target values.

        Returns:
            self: LabelEncoder
                Fitted label encoder.
        """
        self.classes_ = np.unique(y)
        self._label_dict = {label: i for i, label in enumerate(self.classes_)}
        return self

    def transform(self, y: ArrayLike) -> NDArray:
        """
        Transform labels to normalized encoding

        Args:
            y: array-like of shape (n_samples,)
                Target values.
        """
        if self.classes_ is None:
            raise ValueError("LabelEncoder is not fitted. Call 'fit' with appropriate arguments before using this method.")

        y = np.asarray(y)
        return np.array([self._label_dict.get(item, -1) for item in y])

    def fit_transform(self, y: ArrayLike) -> NDArray:
        """
        Fit label encoder and return encoded labels.

        Args:
            y : array-like of shape (n_samples,)
                Target values.

        Returns:
            y_encoded : ndarray of shape (n_samples,)
                Encoded labels.
        """
        return self.fit(y).transform(y)