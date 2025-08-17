
# Import necessary libraries
import numpy as np  # Not used in this code snippet, consider removing
from gensim.models import Word2Vec, KeyedVectors  # For Word2Vec modeling
import logging  # For logging events and errors
import os  # For working with file paths and directories

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Function to train a Word2Vec model
def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0):
    """
    Trains a Word2Vec model on the given sentences.

    Args:
        sentences (list): A list of lists of words (tokenized sentences).
        vector_size (int): Dimensionality of the word vectors.
        window (int): Maximum distance between the current and predicted word within a sentence.
        min_count (int): Ignores all words with total frequency lower than this.
        workers (int): Use these many worker threads to train the model.
        sg (int): Training algorithm: 0 for CBOW, 1 for Skip-gram.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    # Log a message indicating that the Word2Vec model is being trained
    logger.info("Training Word2Vec model...")

    # Create a Word2Vec model with specified parameters
    model = Word2Vec(
        sentences,  # List of tokenized sentences
        vector_size=vector_size,  # Dimensionality of word vectors
        window=window,  # Maximum distance between current and predicted word
        min_count=min_count,  # Ignore words with frequency lower than this
        workers=workers,  # Number of worker threads
        sg=sg  # Training algorithm (CBOW or Skip-gram)
    )

    # Log a message indicating that the Word2Vec model training is complete
    logger.info("Word2Vec model training complete.")

    # Return the trained model
    return model

# Function to load the pre-trained Google News Word2Vec model
def load_google_news_word2vec(model_path="data/GoogleNews-vectors-negative300.bin"):
    """
    Loads the pre-trained Google News Word2Vec model.

    Args:
        model_path (str): Path to the Google News Word2Vec binary file.

    Returns:
        gensim.models.KeyedVectors: The loaded Word2Vec KeyedVectors model.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        # Log an error message if the model file does not exist
        logger.error(f"Google News Word2Vec model not found at {model_path}. Please download it.")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Log a message indicating that the Google News Word2Vec model is being loaded
    logger.info(f"Loading Google News Word2Vec model from {model_path}...")

    # Load the pre-trained Word2Vec model
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Log a message indicating that the Google News Word2Vec model has been loaded
    logger.info("Google News Word2Vec model loaded.")

    # Return the loaded model
    return model

# Function to average Word2Vec embeddings for a list of words
def average_word_embeddings(words, model, vector_size):
    """
    Averages the Word2Vec embeddings for a list of words. Handles out-of-vocabulary (OOV) words by ignoring them.

    Args:
        words (list): A list of words (tokens).
        model (gensim.models.Word2Vec or gensim.models.KeyedVectors): The Word2Vec model.
        vector_size (int): The dimensionality of the word vectors.

    Returns:
        numpy.ndarray: The averaged word embedding vector.
    """
    # Initialize a feature vector with zeros
    feature_vector = np.zeros((vector_size,), dtype="float64")

    # Initialize a counter for the number of words
    num_words = 0

    # Iterate through each word in the list
    for word in words:
        # Check if the word is in the model's vocabulary
        if word in model.wv:
            # Add the word's vector to the feature vector
            feature_vector = np.add(feature_vector, model.wv[word])

            # Increment the word counter
            num_words += 1

    # Check if there are any words
    if num_words:
        # Divide the feature vector by the number of words to get the average
        feature_vector = np.divide(feature_vector, num_words)

    # Return the averaged feature vector
    return feature_vector

# Function to convert a list of documents into a matrix of averaged word embeddings
def get_document_vectors(documents, model, vector_size):
    """
    Converts a list of documents (list of tokens) into a matrix of averaged word embeddings.

    Args:
        documents (list): A list of documents, where each document is a list of tokens.
        model (gensim.models.Word2Vec or gensim.models.KeyedVectors): The Word2Vec model.
        vector_size (int): The dimensionality of the word vectors.

    Returns:
        numpy.ndarray: A 2D numpy array where each row is the averaged embedding for a document.
    """
    # Log a message indicating that document vectors are being generated
    logger.info("Generating document vectors...")

    # Initialize a matrix to store the document vectors
    document_vectors = np.zeros((len(documents), vector_size))

    # Iterate through each document
    for i, doc in enumerate(documents):
        # Get the averaged word embedding for the document
        document_vectors[i] = average_word_embeddings(doc, model, vector_size)

    # Log a message indicating that document vector generation is complete
    logger.info("Document vector generation complete.")

    # Return the matrix of document vectors
    return document_vectors

if __name__ == '__main__':
    # Example usage and testing of Word2Vec utility functions
    from text_preprocessing import preprocess_text
    from load_data import load_imdb_dataset, balance_dataset, split_dataset

    # Define project directories
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    IMDB_DIR = os.path.join(DATA_DIR, "aclImdb")
    GOOGLE_NEWS_MODEL_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin")

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data for Word2Vec testing...")
        full_df = load_imdb_dataset(IMDB_DIR)
        balanced_df = balance_dataset(full_df)
        X_train, X_test, y_train, y_test = split_dataset(balanced_df)

        # Preprocess reviews
        X_train_processed = [preprocess_text(review) for review in X_train]
        X_test_processed = [preprocess_text(review) for review in X_test]

        logger.info("Data preprocessing complete.")

        # Test custom Word2Vec training
        logger.info("\nTesting custom Word2Vec training...")
        custom_w2v_model = train_word2vec_model(X_train_processed, vector_size=100)

        # Log the vocabulary size and vector for a sample word
        logger.info(f"Vocabulary size of custom model: {len(custom_w2v_model.wv.index_to_key)}")
        logger.info(f"Vector for 'movie': {custom_w2v_model.wv['movie'][:5]}...")

        # Test Google News Word2Vec loading
        logger.info("\nTesting Google News Word2Vec loading...")
        try:
            # Load the pre-trained Google News Word2Vec model
            google_w2v_model = load_google_news_word2vec(GOOGLE_NEWS_MODEL_PATH)

            # Log the vector for a sample word
            logger.info(f"Vector for 'movie' from Google News: {google_w2v_model['movie'][:5]}...")

            # Get the vector size of the Google News model
            google_w2v_vector_size = google_w2v_model.vector_size

        except FileNotFoundError:
            # Log a warning message if the model file is not found
            logger.warning("Skipping Google News Word2Vec tests as model file was not found.")

            # Set the model to None
            google_w2v_model = None

        # Test document vector generation
        if custom_w2v_model:
            # Log a message indicating that document vector generation is being tested with the custom model
            logger.info("\nTesting document vector generation with custom model...")

            # Generate document vectors using the custom Word2Vec model
            train_vectors_custom = get_document_vectors(X_train_processed, custom_w2v_model, 100)

            # Log the shape and first few values of the document vectors
            logger.info(f"Shape of custom train vectors: {train_vectors_custom.shape}")
            logger.info(f"First 5 values of first custom document vector: {train_vectors_custom[0, :5]}...")

        if google_w2v_model:
            # Log a message indicating that document vector generation is being tested with the Google News model
            logger.info("\nTesting document vector generation with Google News model...")

            # Generate document vectors using the Google News Word2Vec model
            train_vectors_google = get_document_vectors(X_train_processed, google_w2v_model, google_w2v_vector_size)

            # Log the shape and first few values of the document vectors
            logger.info(f"Shape of Google News train vectors: {train_vectors_google.shape}")
            logger.info(f"First 5 values of first Google News document vector: {train_vectors_google[0, :5]}...")

    except Exception as e:
        # Log any errors that occur during execution
        logger.error(f"Error during Word2Vec utility testing: {e}")
        raise


