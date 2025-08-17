# Import necessary libraries
import os  # For working with file paths and directories
import gensim.downloader as api  # For downloading pre-trained Word2Vec models
import logging  # For logging events and errors

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Function to download the pre-trained Google News Word2Vec model
def download_google_news_word2vec(save_path="data/GoogleNews-vectors-negative300.bin"):
    """
    Downloads the pre-trained Google News Word2Vec model.

    Args:
        save_path (str): The file path where the Word2Vec model will be saved.
    """
    # Check if the model already exists at the specified path
    if os.path.exists(save_path):
        logger.info(f"Google News Word2Vec model already exists at {save_path}. Skipping download.")
        return

    # Log a message indicating that the model is being downloaded
    logger.info("Google News Word2Vec model not found locally. Downloading...")

    try:
        # Ensure the directory exists before saving the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Download the Google News Word2Vec model
        # This is a large model (approx. 3.6 GB), so it will take time.
        model = api.load("word2vec-google-news-300")

        # Save the model to the specified path in binary format
        model.save_word2vec_format(save_path, binary=True)

        # Log a message indicating that the model has been downloaded and saved
        logger.info(f"Google News Word2Vec model downloaded and saved to {save_path}")
    except Exception as e:
        # Log an error message if there's an issue downloading or saving the model
        logger.error(f"Failed to download or save Google News Word2Vec model: {e}")
        raise

if __name__ == '__main__':
    # Get the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define the data directory and the path to the Google News Word2Vec model
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    GOOGLE_NEWS_MODEL_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")

    # Call the function to download the Google News Word2Vec model
    download_google_news_word2vec(GOOGLE_NEWS_MODEL_PATH)