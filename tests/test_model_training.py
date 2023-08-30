import unittest
from src.model_training import build_and_train_model  # Adjust the import path as needed
from src.data_preprocessing import load_and_preprocess_data #
class TestModelTraining(unittest.TestCase):

    def test_model_build_and_training(self):
        # Load test data or mock data
        X_train, X_test, y_train, y_test = load_and_preprocess_data('data/sensor_data.csv')
        X_val, y_val = X_test, y_test  # using test set as validation set for demonstration
        
        config = {'epochs': 1, 'batch_size': 32}
        
        model, history = build_and_train_model(X_train, y_train, X_val, y_val, config)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(history)
