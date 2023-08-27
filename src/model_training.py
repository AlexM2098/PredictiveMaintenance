from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_and_train_model(X_train, y_train, X_val, y_val, config):
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    logger.info("Building and training the model")

    # Initialize the model
    model = Sequential()

    # Add layers
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )

    logger.info("Model training completed")

    return model, history

def evaluate_and_save_model(model, X_test, y_test):
    logger.info("Evaluating and saving the model")

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the model
    model.save("trained_model.h5")

    logger.info("Model saved successfully")
