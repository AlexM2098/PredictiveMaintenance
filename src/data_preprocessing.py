import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the scaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(config):
    filepath = config["data_file_path"]
    test_size = config["test_size"]
    random_state = config["random_state"]

    logger.info(f"Loading data from {filepath}")

    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Data cleaning (here just removing rows with any missing values for demonstration)
    df.dropna(inplace=True)

    # Data Transformation (Standardization of numerical fields)
    scaler = StandardScaler()
    columns_to_scale = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Save the scaler for future use
    joblib.dump(scaler, 'standard_scaler.pkl')

    # Data Splitting
    X = df[columns_to_scale]
    y = df['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logger.info("Data preprocessing completed")

    return X_train, X_test, y_train, y_test
