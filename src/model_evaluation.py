# Import necessary libraries
from nltk.lm import models  # Not used in this code snippet, consider removing
from sklearn.metrics import (
    accuracy_score,  # For calculating accuracy
    precision_score,  # For calculating precision
    recall_score,  # For calculating recall
    f1_score,  # For calculating F1-score
    roc_auc_score,  # For calculating ROC-AUC score
    roc_curve,  # For calculating ROC curve
    confusion_matrix  # Not used in this code snippet, consider removing
)
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # Not used in this code snippet, consider removing
import numpy as np  # Not used in this code snippet, consider removing
import logging  # For logging events and errors
import os  # For working with file paths and directories

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Function to evaluate a given model and return various performance metrics
def evaluate_model(model, X_test, y_test):
    """
    Evaluates a given model and returns various performance metrics.

    Args:
        model: The trained machine learning model.
        X_test (numpy.ndarray): Test feature vectors.
        y_test (pandas.Series): True labels for the test set.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, f1_score, and roc_auc.
    """
    # Make predictions using the model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)  # Accuracy
    precision = precision_score(y_test, predictions)  # Precision
    recall = recall_score(y_test, predictions)  # Recall
    f1 = f1_score(y_test, predictions)  # F1-score
    roc_auc = roc_auc_score(y_test, probabilities)  # ROC-AUC score

    # Store metrics in a dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    # Log the metrics
    logger.info(f"Model Evaluation: {metrics}")

    return metrics

# Function to plot ROC curves for multiple models
def plot_roc_curve(models, X_test, y_test, title="ROC Curve", save_path=None):
    """
    Plots ROC curves for multiple models.

    Args:
        models (dict): A dictionary where keys are model names and values are trained models.
        X_test (numpy.ndarray): Test feature vectors.
        y_test (pandas.Series): True labels for the test set.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Iterate through models and plot ROC curves
    for model_name, model in models.items():
        # Get predicted probabilities
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate ROC curve and AUC score
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    # Set plot labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)

    # Add legend and grid
    ax.legend(loc='lower right')
    ax.grid(True)

    # Tighten layout
    fig.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"ROC curve saved to {save_path}")

    return fig

# Function to plot the confusion matrix for a given model
def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix", save_path=None):
    """
    Plots the confusion matrix for a given model.

    Args:
        model: The trained machine learning model.
        X_test (numpy.ndarray): Test feature vectors.
        y_test (pandas.Series): True labels for the test set.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    # Make predictions using the model
    predictions = model.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the confusion matrix as a heatmap
    sns.heatmap(
        cm,  # Confusion matrix
        annot=True,  # Display numbers in each cell
        fmt='d',  # Format numbers as integers
        cmap='Blues',  # Color scheme
        cbar=False,  # Don't display color bar
        xticklabels=['Negative', 'Positive'],  # Labels for x-axis
        yticklabels=['Negative', 'Positive'],  # Labels for y-axis
        ax=ax  # Axis to plot on
    )

    # Set labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    # Tighten layout
    fig.tight_layout()

    # Save plot if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    return fig


# Main execution block
if __name__ == '__main__':
    # Import necessary functions from other modules
    from load_data import load_imdb_dataset, balance_dataset, split_dataset
    from text_preprocessing import preprocess_text
    from word2vec_utils import train_word2vec_model, get_document_vectors
    from model_training import train_logistic_regression, train_random_forest, save_model, load_model

    # Define project directories
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    IMDB_DIR = os.path.join(DATA_DIR, "aclImdb")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data for model evaluation...")
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

        # Load models for evaluation
        loaded_lr_model = load_model(os.path.join(MODELS_DIR, "logistic_regression.pkl"))  # Load logistic regression model
        loaded_rf_model = load_model(os.path.join(MODELS_DIR, "random_forest.pkl"))  # Load random forest model

        # Evaluate models
        logger.info("\nEvaluating Logistic Regression Model...")
        lr_metrics = evaluate_model(loaded_lr_model, X_test_vectors, y_test)  # Evaluate logistic regression model
        plot_confusion_matrix(loaded_lr_model, X_test_vectors, y_test, title="Logistic Regression Confusion Matrix", save_path=os.path.join(RESULTS_DIR, "lr_confusion_matrix.png"))  # Plot confusion matrix for logistic regression model

        logger.info("\nEvaluating Random Forest Model...")
        rf_metrics = evaluate_model(loaded_rf_model, X_test_vectors, y_test)  # Evaluate random forest model
        plot_confusion_matrix(loaded_rf_model, X_test_vectors, y_test, title="Random Forest Confusion Matrix", save_path=os.path.join(RESULTS_DIR, "rf_confusion_matrix.png"))  # Plot confusion matrix for random forest model

        # Plot ROC curves for both models
        models_to_plot = {"Logistic Regression": loaded_lr_model, "Random Forest": loaded_rf_model}
        plot_roc_curve(models_to_plot, X_test_vectors, y_test, title="ROC Curves for Logistic Regression and Random Forest", save_path=os.path.join(RESULTS_DIR, "roc_curves.png"))  # Plot ROC curves

        logger.info("\nModel evaluation complete. Plots saved to results/ directory.")

    except Exception as e:
        logger.error(f"Error during model evaluation testing: {e}")  # Log any errors that occur during execution
        raise


