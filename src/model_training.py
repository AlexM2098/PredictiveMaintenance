import logging
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from kerastuner import HyperModel, RandomSearch
from scikeras.wrappers import KerasClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_and_train_model(X_train, y_train, X_val, y_val, config):
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
    try:
        history = model.fit(
            X_train, y_train,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            validation_data=(X_val, y_val),
            verbose=1
        )
    except Exception as e:
        logger.error(f"Failed to train the model: {e}")
        return None, None

    logger.info("Model training completed")

    return model, history
