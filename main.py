import json
from src.data_preprocessing import load_and_preprocess_data  # Update this import based on your project structure
from src.model_training import build_and_train_model, evaluate_and_save_model, compute_feature_importance  # Update this import

if __name__ == "__main__":
    # Load configuration settings
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config.get('data_path'))

    # For demonstration, we'll use the test set as a validation set.
    # In a real-world scenario, you should split your data into training, validation, and test sets.
    X_val, y_val = X_test, y_test

    # Step 2: Build and train the model
    model = build_and_train_model(X_train, y_train, X_val, y_val, config)

    # Step 2.5: Compute and display feature importance (Optional but recommended)
    compute_feature_importance(model, X_val, y_val)

    # Step 3: Evaluate and save the model
    evaluate_and_save_model(model, X_test, y_test)
