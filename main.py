import json
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import build_and_train_model, evaluate_and_save_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load configuration settings
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config['data_path'])
  
    # Using the test set as a validation set for demonstration
    X_val, y_val = X_test, y_test

    # Step 2: Build and train the model
    model, history = build_and_train_model(X_train, y_train, X_val, y_val, config)
  
    # Step 3: Evaluate and save the model
    if model and history:
        evaluate_and_save_model(model, X_test, y_test)
