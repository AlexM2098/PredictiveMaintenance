import logging
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from kerastuner import HyperModel, RandomSearch
from keras.wrappers.scikit_learn import KerasClassifier
import eli5
from eli5.sklearn import PermutationImportance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyHyperModel(HyperModel):
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                        activation='relu',
                        input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        return model

def build_and_train_model(X_train, y_train, X_val, y_val, config):
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    logger.info("Building and training the model")

    hypermodel = MyHyperModel(input_shape=(X_train.shape[1],))

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        directory='keras_tuner_dir'
    )

    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Retrieve the best model
    model = tuner.get_best_models(1)[0]

    logger.info("Model training completed")
    
    return model

def evaluate_and_save_model(model, X_test, y_test):
    logger.info("Evaluating and saving the model")

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Save the model
    model.save("trained_model.h5")

    logger.info("Model saved successfully")

def compute_feature_importance(model, X_val, y_val):
    logger.info("Computing feature importance")

    # Wrap the Keras model with KerasClassifier
    model_wrapped = KerasClassifier(build_fn=lambda: model, epochs=0, batch_size=0)

    # Fit PermutationImportance
    perm = PermutationImportance(model_wrapped, random_state=1).fit(X_val, y_val)

    # Display feature importances
    logger.info(eli5.format_as_text(eli5.explain_weights(perm, feature_names=X_val.columns.tolist())))

# Update  main function or wherever  call build_and_train_model
if __name__ == "__main__":
   
    model = build_and_train_model(X_train, y_train, X_val, y_val, config)

    # Add this line to compute feature importance
    compute_feature_importance(model, X_val, y_val)
