# Import necessary libraries
import os  # For working with file paths and directories
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
import logging  # For logging events and errors

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Function to load the IMDb movie review dataset
def load_imdb_dataset(data_dir="data/aclImdb"):
    """
    Loads the IMDb movie review dataset from the specified directory.

    Args:
        data_dir (str): The base directory where the IMDb dataset is located.

    Returns:
        pd.DataFrame: A DataFrame containing 'review' and 'sentiment' columns.
    """
    data = []  # Initialize an empty list to store the data

    # Iterate through 'train' and 'test' subfolders
    for subfolder in ["train", "test"]:
        subfolder_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(subfolder_path):
            logger.warning(f"Subfolder not found: {subfolder_path}")
            continue

        # Iterate through 'pos' and 'neg' sentiment subfolders within train/test
        for sentiment in ["pos", "neg"]:
            sentiment_path = os.path.join(subfolder_path, sentiment)
            if not os.path.exists(sentiment_path):
                logger.warning(f"Subfolder not found: {sentiment_path}")
                continue

            # Iterate through files in the sentiment subfolder
            for file_name in os.listdir(sentiment_path):
                file_path = os.path.join(sentiment_path, file_name)
                try:
                    # Read the file and append the review and sentiment to the data list
                    with open(file_path, "r", encoding="utf-8") as f:
                        data.append([f.read(), 1 if sentiment == "pos" else 0])
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")

    # Check if any data was loaded
    if not data:
        logger.error(f"No data loaded from {data_dir}. Please ensure the dataset is correctly structured.")
        raise ValueError("No valid review files found in the specified directory")

    # Return the loaded data as a DataFrame
    return pd.DataFrame(data, columns=["review", "sentiment"])

# Function to balance the dataset by undersampling the majority class
def balance_dataset(df):
    """
    Balances the dataset by undersampling the majority class.

    Args:
        df (pd.DataFrame): The input DataFrame with 'review' and 'sentiment' columns.

    Returns:
        pd.DataFrame: The balanced DataFrame.
    """
    # Split the DataFrame into positive and negative sentiment reviews
    pos_df = df[df["sentiment"] == 1]
    neg_df = df[df["sentiment"] == 0]

    # Determine the minimum number of samples between the two classes
    min_samples = min(len(pos_df), len(neg_df))

    # Undersample the majority class
    balanced_pos_df = pos_df.sample(n=min_samples, random_state=42)
    balanced_neg_df = neg_df.sample(n=min_samples, random_state=42)

    # Combine the balanced DataFrames and shuffle the rows
    return pd.concat([balanced_pos_df, balanced_neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Function to split the dataset into training and testing sets
def split_dataset(df, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: X_train, X_test, y_train, y_test DataFrames.
    """
    # Split the DataFrame into reviews and sentiments
    X = df["review"]
    y = df["sentiment"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the data directory and the path to the IMDb dataset
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    IMDB_DIR = os.path.join(DATA_DIR, "aclImdb")

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if the IMDb dataset exists
    if not os.path.exists(IMDB_DIR):
        logger.error(
            f"IMDb dataset not found at {IMDB_DIR}. Please download and extract the dataset from: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        exit(1)

    try:
        # Load datasets
        logger.info("Loading IMDb datasets...")
        full_df = load_imdb_dataset(IMDB_DIR)
        logger.info(f"Loaded {len(full_df)} reviews.")

        # Balance dataset
        balanced_df = balance_dataset(full_df)
        logger.info(f"Balanced dataset has {len(balanced_df)} reviews.")

        # Split dataset
        X_train, X_test, y_train, y_test = split_dataset(balanced_df)
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise