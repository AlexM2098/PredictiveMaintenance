import unittest
from src.data_preprocessing import load_and_preprocess_data
import numpy as np

class TestDataPreprocessing(unittest.TestCase):

    def test_load_and_preprocess_data(self):
        # Note: Update the path to the actual data location
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data/sensor_data.csv')
        
        # Check if data is loaded
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

        # Check the shape of the data
        # Assuming there are 5 features in the dataset
        self.assertEqual(X_train.shape[1], 5)
        self.assertEqual(X_test.shape[1], 5)
        
        # Check if data is scaled
        # Mean should be close to 0 and standard deviation close to 1 for each feature
        print("Means of features: ", np.mean(X_train, axis=0))

        self.assertTrue(np.allclose(np.mean(X_train, axis=0), 0, atol=1e-2))


        self.assertTrue(np.allclose(np.std(X_train, axis=0), 1, atol=1e-2))

if __name__ == '__main__':
    unittest.main()
