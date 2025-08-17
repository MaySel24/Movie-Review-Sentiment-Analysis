# Import necessary libraries
from sklearn.linear_model import LogisticRegression  # For logistic regression classification
from sklearn.ensemble import RandomForestClassifier  # For random forest classification (not used in this function)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score  # For model evaluation (not used in this function)
import joblib  # For model serialization (not used in this function)
import logging  # For logging events and errors
import os  # For working with file paths and directories (not used in this function)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Function to train a logistic regression model
def train_logistic_regression(X_train, y_train, random_state=40):
    """
    Trains a Logistic Regression classifier.

    Args:
        X_train (numpy.ndarray): Training feature vectors.
        y_train (pandas.Series): Training labels.
        random_state (int): Random state for reproducibility.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained Logistic Regression model.
    """
    # Log a message indicating that the logistic regression model is being trained
    logger.info("Training Logistic Regression model...")

    # Create a logistic regression model with a specified random state and maximum number of iterations
    model = LogisticRegression(random_state=random_state, max_iter=1000)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Log a message indicating that the logistic regression model training is complete
    logger.info("Logistic Regression model training complete.")

    # Return the trained model
    return model


# Function to train a random forest model
def train_random_forest(X_train, y_train, n_estimators=500, random_state=40):
    """
    Trains a Random Forest classifier.

    Args:
        X_train (numpy.ndarray): Training feature vectors.
        y_train (pandas.Series): Training labels.
        n_estimators (int): Number of trees in the forest.
        random_state (int): Random state for reproducibility.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained Random Forest model.
    """
    # Log a message indicating that the random forest model is being trained
    logger.info("Training Random Forest model...")

    # Create a random forest model with a specified number of estimators and random state
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Train the model using the training data
    model.fit(X_train, y_train)

    # Log a message indicating that the random forest model training is complete
    logger.info("Random Forest model training complete.")

    # Return the trained model
    return model

# Function to save a trained model to disk
def save_model(model, model_path):
    """
    Saves a trained model to disk.

    Args:
        model: The trained model to save.
        model_path (str): The file path where the model will be saved.
    """
    # Ensure the directory exists before saving the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Use joblib to serialize and save the model
    joblib.dump(model, model_path)

    # Log a message indicating that the model has been saved
    logger.info(f"Model saved to {model_path}")

# Function to load a trained model from disk
def load_model(model_path):
    """
    Loads a trained model from disk.

    Args:
        model_path (str): The file path where the model is saved.

    Returns:
        The loaded model.
    """
    # Check if the model file exists
    if not os.path.exists(model_path):
        # Log an error message if the model file does not exist
        logger.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Use joblib to deserialize and load the model
    model = joblib.load(model_path)

    # Log a message indicating that the model has been loaded
    logger.info(f"Model loaded from {model_path}")

    # Return the loaded model
    return model

# Main execution block
if __name__ == '__main__':
    # Import necessary functions from other modules
    from load_data import load_imdb_dataset, balance_dataset, split_dataset
    from text_preprocessing import preprocess_text
    from word2vec_utils import train_word2vec_model, get_document_vectors

    # Define project directories
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    IMDB_DIR = os.path.join(DATA_DIR, "aclImdb")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data for model training...")
        full_df = load_imdb_dataset(IMDB_DIR)  # Load IMDb dataset
        balanced_df = balance_dataset(full_df)  # Balance dataset
        X_train, X_test, y_train, y_test = split_dataset(balanced_df)  # Split dataset into training and testing sets

        # Preprocess reviews
        X_train_processed = [preprocess_text(review) for review in X_train]  # Preprocess training reviews
        X_test_processed = [preprocess_text(review) for review in X_test]  # Preprocess testing reviews

        # Train Word2Vec model
        logger.info("Training Word2Vec model...")
        w2v_model = train_word2vec_model(X_train_processed, vector_size=100)  # Train Word2Vec model on training reviews

        # Generate document vectors
        X_train_vectors = get_document_vectors(X_train_processed, w2v_model, 100)  # Get document vectors for training reviews
        X_test_vectors = get_document_vectors(X_test_processed, w2v_model, 100)  # Get document vectors for testing reviews

        # Train models
        lr_model = train_logistic_regression(X_train_vectors, y_train)  # Train logistic regression model
        rf_model = train_random_forest(X_train_vectors, y_train)  # Train random forest model

        # Save models
        save_model(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))  # Save logistic regression model
        save_model(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))  # Save random forest model

        # Test loading models
        loaded_lr_model = load_model(os.path.join(MODELS_DIR, "logistic_regression.pkl"))  # Load logistic regression model
        loaded_rf_model = load_model(os.path.join(MODELS_DIR, "random_forest.pkl"))  # Load random forest model

        # Make predictions
        lr_predictions = loaded_lr_model.predict(X_test_vectors)  # Make predictions using logistic regression model
        rf_predictions = loaded_rf_model.predict(X_test_vectors)  # Make predictions using random forest model

        # Evaluate models
        lr_accuracy = accuracy_score(y_test, lr_predictions)  # Calculate accuracy of logistic regression model
        rf_accuracy = accuracy_score(y_test, rf_predictions)  # Calculate accuracy of random forest model

        # Log accuracy of models
        logger.info(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
        logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    except Exception as e:
        # Log any errors that occur during execution
        logger.error(f"Error during model training testing: {e}")
        raise

