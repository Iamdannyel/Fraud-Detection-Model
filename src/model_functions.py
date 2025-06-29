import logging
import pandas as pd
import os
import numpy as np

#set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean dataset, dropping 'isFlaggedFraud', supporting CSV and Parquet files.
    
    Args:
        file_path (str): Path to the dataset (CSV or Parquet).
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If 'isFraud' column is missing or file format is unsupported.
    """

    logger = logging.getLogger(__name__)

    # check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Dataset not found at {file_path}")
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            logger.error(f"Unsupported file format: {file_extension}. Use CSV or Parquet.")
            raise ValueError(f"Unsupported file format: {file_extension}. Use CSV or Parquet.")
        
        logger.info(f"Loaded data from {file_path} with {len(df)} rows.")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

    # drop 'isFlaggedFraud' if present
    try:
        if 'isFlaggedFraud' in df.columns:
            df = df.drop('isFlaggedFraud', axis=1)
            logger.info("Dropped 'isFlaggedFraud' column.")
        else:
            logger.info("'isFlaggedFraud' column not found in dataset.")
    except Exception as e:
        logger.error(f"Error dropping 'isFlaggedFraud': {str(e)}")
        raise

    # validate 'isFraud' column
    if 'isFraud' not in df.columns:
        logger.error("Target column 'isFraud' missing")
        raise ValueError("Target column 'isFraud' missing")

    return df


def stratified_subsample(df: pd.DataFrame, frac: float, target_col: str, random_state: int = 42) -> pd.DataFrame:
    """
    Subsample DataFrame, keeping all fraud cases and sampling non-fraud cases.

    Args:
        df (pd.DataFrame): Input DataFrame.
        frac (float): Fraction of non-fraud data to sample (0 < frac <= 1).
        target_col (str): Column name for stratification (e.g., 'isFraud').
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Subsampled DataFrame with all fraud cases and sampled non-fraud cases.

    Raises:
        ValueError: If frac is invalid, target_col is missing, or no fraud cases exist.
    """
    if not 0 < frac <= 1:
        raise ValueError("frac must be between 0 and 1")
    if target_col not in df.columns:
        raise ValueError(f"Column {target_col} not found in DataFrame")

    logging.info(f"Subsampling {frac*100}% of non-fraud data, keeping all fraud cases...")

    df_fraud = df[df[target_col] == 1]
    df_non_fraud = df[df[target_col] == 0]

    # validate fraud cases
    if df_fraud.empty:
        raise ValueError("No fraud cases found in the dataset")

    # sample non-fraud cases
    df_non_fraud_sample = df_non_fraud.sample(frac=frac, random_state=random_state)

    df_sample = pd.concat([df_fraud, df_non_fraud_sample], ignore_index=True)

    return df_sample



def engineer_features(df, window_size=6):
    """
    Perform feature engineering on a transaction dataset to prepare it for fraud detection.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns 'step', 'amount', 'nameOrig', 'isFraud', etc.
    - window_size (int): Size of the rolling window for transaction count (default=6).
    
    Returns:
    - pd.DataFrame: Processed DataFrame with engineered features.
    
    Raises:
    - ValueError: If required columns are missing or data types are invalid.
    - TypeError: If input is not a pandas DataFrame.
    """
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame")
        
        required_columns = ['step', 'amount', 'nameOrig', 'isFraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if not pd.api.types.is_numeric_dtype(df['step']) or not pd.api.types.is_numeric_dtype(df['amount']):
            raise ValueError("Columns 'step' and 'amount' must be numeric")

        df = df.sort_values('step').reset_index(drop=True)
        logger.info("Sorted DataFrame by 'step'")

        df['log_amount'] = np.log1p(df['amount'])
        logger.info("Applied log transformation to 'amount'")

        if window_size < 1:
            raise ValueError("window_size must be a positive integer")
        df['tx_count_6hr'] = (
            df.groupby('nameOrig')['step']
              .rolling(window=window_size, min_periods=1)
              .count()
              .reset_index(level=0, drop=True)
        )
        logger.info(f"Computed rolling transaction count with window size {window_size}")

        df['cum_amount_sent'] = df.groupby('nameOrig')['log_amount'].cumsum()
        logger.info("Computed cumulative amount sent by sender")

        df = df.sort_values(['nameOrig', 'step'])  # sort for sender-specific features
        logger.info("Sorted DataFrame by 'nameOrig' and 'step'")
        df['sender_mean'] = (
            df.groupby('nameOrig')['log_amount']
              .expanding()
              .mean()
              .shift(1)
              .reset_index(level=0, drop=True)
        )
        df['sender_std'] = (
            df.groupby('nameOrig')['log_amount']
              .expanding()
              .std()
              .shift(1)
              .reset_index(level=0, drop=True)
        )
        df['sender_std'] = df['sender_std'].fillna(1).replace(0, 1)
        logger.info("Computed sender mean, std, and handled zeros")
        
        # identify outlier transaction based on user's history
        df['zscore_amount'] = (df['log_amount'] - df['sender_mean']) / df['sender_std']
        logger.info("Computed Z-score for outlier detection")

        # time since last transaction
        df['prev_step'] = df.groupby('nameOrig')['step'].shift(1)
        df['time_since_last_tx'] = df['step'] - df['prev_step']
        df['is_first_tx'] = df['prev_step'].isna().astype(int)
        df['time_since_last_tx'] = df['time_since_last_tx'].fillna(df['time_since_last_tx'].median())
        logger.info("Computed time since last transaction features")

        # drop temporary column and handle NaN values
        df = df.drop(['prev_step'], axis=1)
        df = df.dropna()
        logger.info("Dropped 'prev_step' and removed NaN values")

        return df

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except TypeError as te:
        logger.error(f"TypeError: {te}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during feature engineering: {e}")
        raise




def identify_repeat_perpetrator(df, df_train, step_col='step', dest_col='nameDest', fraud_col='isFraud'):
    """
    Create a feature flagging repeat perpetrators based on past fraud in training data.
    
    Args:
        df (pd.DataFrame): DataFrame to add feature to (train or test).
        df_train (pd.DataFrame): Training DataFrame for fraud history.
        step_col (str): Column for time step.
        dest_col (str): Column for destination account.
        fraud_col (str): Column for fraud label.
    
    Returns:
        pd.DataFrame: DataFrame with new feature.
    """
    df = df.sort_values(step_col).copy()
    df['is_repeat_perpetrator'] = 0
    df['fraud_count_orig'] = 0
    
    fraud_history = set()
    fraud_counts = {}
    
    # process training data to build history
    for step in df_train[step_col].unique():
        # Fraudulent nameDest before or at current step
        past_fraud = df_train[(df_train[step_col] <= step) & (df_train[fraud_col] == 1)][dest_col]
        fraud_history.update(past_fraud)
        for orig in past_fraud:
            fraud_counts[orig] = fraud_counts.get(orig, 0) + 1
    
    for step in df[step_col].unique():
        mask = df[step_col] == step
        # flag nameDest with prior fraud
        df.loc[mask, 'is_repeat_perpetrator'] = df.loc[mask, dest_col].isin(fraud_history).astype(int)
        # assign fraud count
        df.loc[mask, 'fraud_count_orig'] = df.loc[mask, dest_col].map(fraud_counts).fillna(0).astype(int)
    
    return df