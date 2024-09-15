from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
           
    def test_min_max_scaler(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line

    def test_min_max_scalers_with_constant_feature(self):
        
        data = [[1, 5, 0], [2, 5, 1], [3, 5, 2], [4, 5, 3]]
        
        # Test MinMaxScaler
        mm_scaler = MinMaxScaler()
        mm_result = mm_scaler.fit_transform(data)
        
        # Check if MinMaxScaler handles constant feature (should be NaN)
        assert all(np.isnan(x) for x in mm_result[:, 1]), "MinMaxScaler should return NaN for constant feature"
        
        # Verify other features are scaled correctly
        assert np.allclose(mm_result[:, 0], [0, 1/3, 2/3, 1]), "MinMaxScaler failed to scale non-constant feature correctly"
        assert np.allclose(mm_result[:, 2], [0, 1/3, 2/3, 1]), "MinMaxScaler failed to scale non-constant feature correctly"
        

    def test_std_scalers_with_constant_feature(self): 

        data = [[1, 5, 0], [2, 5, 1], [3, 5, 2], [4, 5, 3]]
        
        # Test StandardScaler
        std_scaler = StandardScaler()
        std_result = std_scaler.fit_transform(data)
        
        # Check if StandardScaler handles constant feature (should be NaN)
        assert all(np.isnan(x) for x in std_result[:, 1]), "StandardScaler should return NaN for constant feature"
        
        # Verify other features are scaled correctly
        assert np.allclose(np.mean(std_result[:, 0]), 0) and np.allclose(np.std(std_result[:, 0]), 1), \
            "StandardScaler failed to scale non-constant feature correctly"
        assert np.allclose(np.mean(std_result[:, 2]), 0) and np.allclose(np.std(std_result[:, 2]), 1), \
            "StandardScaler failed to scale non-constant feature correctly"


class TestLabelEncoder(TestCase):
    def test_initialize_label_encoder(self):
        encoder = LabelEncoder()
        assert isinstance(encoder, LabelEncoder), "encoder is not a Label Encoder object "

    def test_label_encoder_fit(self):
        encoder = LabelEncoder()
        data = ['cat', 'dog', 'bird', 'cat', 'dog']
        expected = np.array(['bird', 'cat', 'dog'])
        encoder.fit(data)
        assert (encoder.classes_ == expected).all(), "encoder fit does not return correct classes"

    def test_label_encoder_transform(self):
        encoder = LabelEncoder()
        data = ['cat', 'dog', 'bird', 'cat', 'dog']
        expected = np.array([1, 2, 0, 1, 2])
        encoder.fit(data)
        result = encoder.transform(data)
        assert (result == expected).all(), f"Encoder transform does not return expected values. Expected: {expected}, Got: {result}"

    def test_label_encoder_fit_transform(self):
        encoder = LabelEncoder()
        data = ['apple', 'banana', 'apple', 'cherry']
        expected = np.array([0, 1, 0, 2])
        result = encoder.fit_transform(data)
        assert (result == expected).all(), f"Encoder fit_transform does not return expected values. Expected: {expected}, Got: {result}"

    def test_label_encoder_transform_unseen_label(self):
        encoder = LabelEncoder()
        data = ['cat', 'dog', 'bird']
        encoder.fit(data)
        result = encoder.transform(['fish', 'dog', 'cat'])
        expected = np.array([-1, 2, 1])
        assert (result == expected).all(), f"Encoder transform does not handle unseen label correctly. Expected: {expected}, Got: {result}"

    def test_label_encoder_with_numeric_labels(self):
        encoder = LabelEncoder()
        data = [1, 2, 3, 1, 2, 3, 4]
        expected = np.array([0, 1, 2, 0, 1, 2, 3])
        result = encoder.fit_transform(data)
        assert (result == expected).all(), f"Encoder fit_transform does not handle numeric labels correctly. Expected: {expected}, Got: {result}"


if __name__ == '__main__':
    unittest.main()