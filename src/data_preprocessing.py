import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the scaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath='data/sensor_data.csv'):
    logger.info("Loading and preprocessing data")

    # Load the dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found at {filepath}")
        return None

    # Data validation (can add more checks)
    if df.empty:
        logger.warning("The CSV file is empty")
        return None
    
    # Data cleaning 
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
