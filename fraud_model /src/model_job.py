import os
import sys
import time
import pickle
import pandas as pd
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_functions import load_data, engineer_features



# configuration 
INPUT_FOLDER = r'C:\Users\chino\Downloads\fraud_model\transactions_data'  # Folder to monitor for new CSV files
MODEL_PATH = r'C:\Users\chino\Downloads\fraud_model\fraud_xgbmodel.pkl'  # Path to the pickle file
OUTPUT_FOLDER = r'C:\Users\chino\Downloads\output_predictions'  # Folder to save prediction results
FEATURE_COLUMNS = None  # List of feature columns used by the model (set during model loading)
CHECK_INTERVAL = 100000  # Seconds to wait if no new files are found (for manual checking)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NewFileHandler(FileSystemEventHandler):
    """Handle new file events in the monitored folder."""
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns

    def on_created(self, event):
        """Triggered when a new file is created."""
        if event.is_directory:
            return
        if event.src_path.endswith('.csv'):
            logger.info(f"New file detected: {event.src_path}")
            process_new_file(event.src_path, self.model, self.feature_columns)

def load_model_and_columns(model_path):
    """Load the model and feature columns from the pickle file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        feature_columns = getattr(model, 'feature_names_in_', None)
        if feature_columns is None:
            logger.info("Warning: Model does not have feature_names_in_. Please specify FEATURE_COLUMNS manually.")
        return model, feature_columns
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        raise

def process_new_file(file_path, model, feature_columns ):
    """Process a new CSV file, engineer features, make predictions, and save results."""
    try:
        # load the new data
        df = load_data(file_path)
        logger.info(f"Loaded data from {file_path} with {len(df)} rows.")

        df = engineer_features(df)

        # ensure required feature columns are present
        if feature_columns is not None:
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.info(f"Error: Missing columns in {file_path}: {missing_cols}")
                return
            X = df[feature_columns]
        else:
            # assume all columns except 'isFraud' (if present) are features
            X = df.drop(columns=['isFraud'], errors='ignore')

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

        # create output dataframe
        output_df = df.copy()
        output_df['predicted_isFraud'] = predictions
        if probabilities is not None:
            output_df['fraud_probability'] = probabilities

        # generate output file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"predictions_{os.path.basename(file_path).replace('.csv', '')}_{timestamp}.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # save predictions
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logger.info(f"Error processing {file_path}: {e}")

def main():
    """Set up the folder monitoring pipeline."""
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    model, feature_columns = load_model_and_columns(MODEL_PATH)

    # set up the observer for the input folder
    event_handler = NewFileHandler(model, feature_columns)
    observer = Observer()
    observer.schedule(event_handler, INPUT_FOLDER, recursive=False)
    observer.start()

    logger.info(f"Monitoring {INPUT_FOLDER} for new CSV files...")
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped monitoring.")
    observer.join()

if __name__ == "__main__":
    main()