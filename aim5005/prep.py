import numpy as np
from typing import List, Tuple

def center(x:List[float]) -> List[float]:
    """
    Subtract the mean so that the result is centered around 0
    """
    mean_val = np.mean(x)
    return [x - mean_val for x in x]

def covariance(x: List[float], y:List[float]) -> float:
    """
    Measures how much two variables vary from their means
    """
    assert len(x) == len(y)
    
    return np.dot(center(x), center(y)) / (len(x) - 1)

def correlation(x: List[float], y:List[float]) -> float:
    """
    A measure of how much variance there is in x and y around the means. 
    The correlation will always be between -1 and 1. 
    1 is perfect correlation and 0 is no correlation at all
    """
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x > 0 and std_y > 0:
        return covariance(x, y)/std_x/std_y
    else:
        return 0


def zscore(x: List[float]) -> List[float]: 
    """
    Compute the z-score for each value in a list of numbers 

    The z-score represents the number of standard deviations by which the value is above the mean.
    A z-score of 0 means the value is equal to the mean, while a z-score of 1 means the value
    is one standard deviation above the mean.
    """

    mean_val = np.mean(x)
    std_val = np.std(x)

    return [(xi - mean_val) / std_val for xi in x] if std_val > 0 else [0] * lean(x)


def summary_statistics(x: List[float]) -> dict:  
    """
    Compute summary statistics for a list of numbers.
    """

    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "std_dev": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "q1": np.percentile(x, 25),
        "q3": np.percentile(x, 75)

    }