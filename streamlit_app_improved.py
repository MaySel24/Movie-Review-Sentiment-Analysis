# Import necessary libraries
import streamlit as st  # For building the web application
import os  # For working with file paths and directories
import sys
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For creating informative and attractive statistical graphics
from collections import Counter  # For counting frequencies of elements
import logging  # For logging events and errors

# Import custom functions from local modules
from src.load_data import (
    load_imdb_dataset,  # Load IMDB dataset
    balance_dataset,  # Balance the dataset for better model performance
    split_dataset  # Split the dataset into training and testing sets
)
from src.text_preprocessing import preprocess_text  # Preprocess text data
from src.word2vec_utils import (
    train_word2vec_model,  # Train a Word2Vec model
    get_document_vectors  # Get document vectors using Word2Vec
)
from src.model_training import (
    train_logistic_regression,  # Train a logistic regression model
    train_random_forest,  # Train a random forest model
    save_model,  # Save a trained model
    load_model  # Load a saved model
)
from src.model_evaluation import (
    evaluate_model,  # Evaluate the performance of a model
    plot_roc_curve,  # Plot the ROC curve
    plot_confusion_matrix  # Plot the confusion matrix
)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# --- Configuration ---
# Define project directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Get the project root directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # Directory for storing data
IMDB_DIR = os.path.join(DATA_DIR, "aclImdb")  # Directory for IMDB dataset
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # Directory for storing models
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")  # Directory for storing results

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)  # Create data directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)  # Create models directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)  # Create results directory if it doesn't exist

# --- Streamlit App Configuration ---
# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",  # Set the page title
    page_icon="🎬",  # Set the page icon
    layout="wide",  # Set the layout to wide
    initial_sidebar_state="expanded"  # Set the initial sidebar state to expanded
)

# Custom CSS for better styling
# Use markdown to add custom CSS styles to the app
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    </style>
""", unsafe_allow_html=True)

# --- Caching functions for performance ---
# Use caching to improve performance by storing the results of expensive function calls
@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """
    Load and preprocess the IMDb dataset.

    Returns:
        tuple: X_train, X_test, y_train, y_test, X_train_processed, X_test_processed
    """
    try:
        # Display a spinner while loading and preprocessing data
        with st.spinner("Loading and preprocessing data..."):
            # Load the IMDb dataset
            full_df = load_imdb_dataset(IMDB_DIR)
            # Balance the dataset
            balanced_df = balance_dataset(full_df)
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = split_dataset(balanced_df)
            # Preprocess the reviews
            X_train_processed = [preprocess_text(review) for review in X_train]
            X_test_processed = [preprocess_text(review) for review in X_test]
            return X_train, X_test, y_train, y_test, X_train_processed, X_test_processed
    except Exception as e:
        # Display an error message if there's an issue loading the data
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the IMDb dataset is available in the data/aclImdb directory.")
        st.stop()

@st.cache_resource(show_spinner=False)
def train_word2vec_and_vectorize(X_train_processed, X_test_processed):
    """
    Train a Word2Vec model and generate document vectors.

    Args:
        X_train_processed (list): List of preprocessed training reviews.
        X_test_processed (list): List of preprocessed testing reviews.

    Returns:
        tuple: w2v_model, X_train_vectors, X_test_vectors
    """
    # Display a spinner while training the Word2Vec model and generating vectors
    with st.spinner("Training Word2Vec model and generating vectors..."):
        # Train a Word2Vec model
        w2v_model = train_word2vec_model(X_train_processed, vector_size=100)
        # Generate document vectors for the training and testing sets
        X_train_vectors = get_document_vectors(X_train_processed, w2v_model, 100)
        X_test_vectors = get_document_vectors(X_test_processed, w2v_model, 100)
        return w2v_model, X_train_vectors, X_test_vectors

@st.cache_resource(show_spinner=False)
def train_and_save_models(X_train_vectors, y_train):
    """
    Train and save machine learning models.

    Args:
        X_train_vectors (array): Training vectors.
        y_train (array): Training labels.

    Returns:
        tuple: lr_model, rf_model
    """
    # The training and saving process can be time-consuming, so display a spinner
    with st.spinner("Training machine learning models..."):
        # Train a logistic regression model
        lr_model = train_logistic_regression(X_train_vectors, y_train)
        # Train a random forest model
        rf_model = train_random_forest(X_train_vectors, y_train)
        # Save the trained models
        save_model(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
        save_model(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))
        return lr_model, rf_model

def predict_sentiment(review_text, model, w2v_model):
    """
        Predict the sentiment of a given review.

        Args:
            review_text (str): The text of the review.
            model: The trained machine learning model.
            w2v_model: The trained Word2Vec model.

        Returns:
            tuple: sentiment, confidence, probabilities
        """
    # Check if the review text is empty
    if not review_text.strip():
        return None, None, None

    # Preprocess the review text
    processed_review = preprocess_text(review_text)

    # Check if the review has valid tokens
    if not processed_review:
        return "Unknown", 0.5, [0.5, 0.5]

    # Generate a vector for the review
    review_vector = get_document_vectors([processed_review], w2v_model, 100)

    # Make a prediction using the trained model
    prediction = model.predict(review_vector)[0]
    probabilities = model.predict_proba(review_vector)[0]

    # Determine the sentiment based on the prediction
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[prediction]

    return sentiment, confidence, probabilities


# --- Main App ---
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("🔧 SIDEBAR")

    # Load data
    try:
        X_train, X_test, y_train, y_test, X_train_processed, X_test_processed = load_and_preprocess_data()
        w2v_model, X_train_vectors, X_test_vectors = train_word2vec_and_vectorize(X_train_processed, X_test_processed)
        lr_model, rf_model = train_and_save_models(X_train_vectors, y_train)
    except Exception as e:
        st.error(f"Failed to initialize the application: {str(e)}")
        st.stop()

    # Navigation
    pages = {
        "🏠 Home": show_home_page,
        "📊 Dataset Overview": show_dataset_overview,
        "🔍 Model Evaluation": show_model_evaluation,
        "🎯 Predict Sentiment": show_prediction_page,
        "ℹ️ About": show_about_page
    }

    page = st.sidebar.selectbox("📚 Navigate to:", list(pages.keys()))

    # Pass relevant arguments to each page function
    if page == "🏠 Home":
        pages[page](X_train, X_test, y_train, y_test, lr_model, rf_model, X_test_vectors, w2v_model)
    elif page == "📊 Dataset Overview":
        pages[page](X_train, X_test, y_train, y_test, X_train_processed)
    elif page == "🔍 Model Evaluation":
        pages[page](lr_model, rf_model, X_test_vectors, y_test)
    elif page == "🎯 Predict Sentiment":
        pages[page](lr_model, rf_model, w2v_model)
    elif page == "ℹ️ About":
        pages[page]()


# Function to display the home page with project overview and quick stats
def show_home_page(X_train, X_test, y_train, y_test, lr_model, rf_model, X_test_vectors, w2v_model):
    """
    Display the home page with project overview and quick stats.
    """
    # Display the header and introduction
    st.header("🏠 Welcome to Movie Review Sentiment Analysis")
    st.markdown("""
        This application demonstrates a complete sentiment analysis pipeline for movie reviews using machine learning.
        Navigate through the different sections to explore the dataset, evaluate models, and make predictions.
    """)

    # Quick stats: Displays key metrics about the dataset and model performance
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric("Total Reviews", f"{len(X_train) + len(X_test):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric("Training Set", f"{len(X_train):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric("Test Set", f"{len(X_test):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        # Quick model performance: Displays the accuracy of the Logistic Regression model
        lr_metrics = evaluate_model(lr_model, X_test_vectors, y_test)
        st.markdown("<div class=\"metric-card\">", unsafe_allow_html=True)
        st.metric("Best Accuracy", f"{lr_metrics['accuracy']:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Quick prediction demo: Allows users to quickly test sentiment prediction
    st.subheader("🚀 Quick Prediction Demo")
    demo_review = st.text_input(
        "Try a quick sentiment prediction:",
        placeholder="Enter a movie review here...",
        value="This movie was absolutely fantastic! Great acting and storyline."
    )
    if demo_review:
        # Predict sentiment using the Logistic Regression model and Word2Vec model
        sentiment, confidence, _ = predict_sentiment(demo_review, lr_model, w2v_model)
        if sentiment:
            emoji = "😊" if sentiment == "Positive" else "😞"
            st.success(f"{emoji} **{sentiment}** (Confidence: {confidence:.1%})")

# Function to display dataset statistics and examples
def show_dataset_overview(X_train, X_test, y_train, y_test, X_train_processed):
    """
    Display dataset statistics and examples.
    """
    # Display the header
    st.header("📊 Dataset Overview")

    # Dataset statistics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Dataset Statistics")
        stats_df = pd.DataFrame({
            "Metric": ["Total Reviews", "Training Reviews", "Test Reviews", "Positive Reviews", "Negative Reviews"],
            "Count": [
                len(X_train) + len(X_test),
                len(X_train),
                len(X_test),
                sum(y_train) + sum(y_test),
                len(y_train) - sum(y_train) + len(y_test) - sum(y_test)
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    with col2:
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = pd.Series(y_train).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#e74c3c', '#2ecc71']
        ax.pie(sentiment_counts.values, labels=['Negative', 'Positive'], autopct='%1.1f%%', colors=colors)
        ax.set_title('Training Set Sentiment Distribution')
        st.pyplot(fig)

    # Sample reviews
    st.subheader("📝 Sample Reviews")
    sample_indices = np.random.choice(len(X_train), 3, replace=False)
    for i, idx in enumerate(sample_indices):
        sentiment_label = "Positive" if y_train.iloc[idx] == 1 else "Negative"
        emoji = "😊" if sentiment_label == "Positive" else "😞"
        with st.expander(f"{emoji} Sample {i + 1}: {sentiment_label} Review"):
            st.write("**Original Review:**")
            st.write(X_train.iloc[idx][:500] + "..." if len(X_train.iloc[idx]) > 500 else X_train.iloc[idx])
            st.write("**Preprocessed Tokens:**")
            st.write(X_train_processed[idx][:20])

# Function to display detailed model evaluation results
def show_model_evaluation(lr_model, rf_model, X_test_vectors, y_test):
    """
    Display detailed model evaluation results.
    """
    # Display the header
    st.header("🔍 Model Evaluation")

    # Evaluate both models
    lr_metrics = evaluate_model(lr_model, X_test_vectors, y_test)
    rf_metrics = evaluate_model(rf_model, X_test_vectors, y_test)

    # Performance comparison
    st.subheader("📊 Performance Comparison")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Logistic Regression": [
            f"{lr_metrics['accuracy']:.3f}",
            f"{lr_metrics['precision']:.3f}",
            f"{lr_metrics['recall']:.3f}",
            f"{lr_metrics['f1_score']:.3f}",
            f"{lr_metrics['roc_auc']:.3f}"
        ],
        "Random Forest": [
            f"{rf_metrics['accuracy']:.3f}",
            f"{rf_metrics['precision']:.3f}",
            f"{rf_metrics['recall']:.3f}",
            f"{rf_metrics['f1_score']:.3f}",
            f"{rf_metrics['roc_auc']:.3f}"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Confusion Matrices")
        # Logistic Regression Confusion Matrix
        st.write("**Logistic Regression**")
        fig_lr = plot_confusion_matrix(lr_model, X_test_vectors, y_test, title="Logistic Regression Confusion Matrix")
        st.pyplot(fig_lr)
        # Random Forest Confusion Matrix
        st.write("**Random Forest**")
        fig_rf = plot_confusion_matrix(rf_model, X_test_vectors, y_test, title="Random Forest Confusion Matrix")
        st.pyplot(fig_rf)
    with col2:
        st.subheader("📈 ROC Curves")
        models_dict = {"Logistic Regression": lr_model, "Random Forest": rf_model}
        fig_roc = plot_roc_curve(models_dict, X_test_vectors, y_test, title="ROC Curves Comparison")
        st.pyplot(fig_roc)

    # Performance insights
    st.subheader("💡 Key Insights")
    better_model = "Logistic Regression" if lr_metrics['accuracy'] > rf_metrics['accuracy'] else "Random Forest"
    st.info(f"🏆 **{better_model}** performs better with higher accuracy.")
    if lr_metrics['roc_auc'] > 0.8:
        st.success("🎯 Both models show excellent performance (ROC-AUC > 0.8)")
    elif lr_metrics['roc_auc'] > 0.7:
        st.info("👍 Both models show good performance (ROC-AUC > 0.7)")


# Function to display the sentiment prediction interface
def show_prediction_page(lr_model, rf_model, w2v_model):
    """
    Display the sentiment prediction interface.
    """
    # Display the header
    st.header("🎯 Predict Sentiment")

    # Model selection
    model_choice = st.selectbox(
        "🤖 Choose Model:",
        ["Logistic Regression", "Random Forest"],
        help="Select which trained model to use for prediction"
    )
    selected_model = lr_model if model_choice == "Logistic Regression" else rf_model

    # Input area
    st.subheader("📝 Enter Your Movie Review")
    user_review = st.text_area(
        "Type your movie review here:",
        height=150,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I would definitely recommend it to anyone who enjoys great storytelling.",
        help="Enter a movie review to analyze its sentiment"
    )

    # Prediction button
    if st.button("🔮 Predict Sentiment", type="primary"):
        if user_review.strip():
            with st.spinner("Analyzing sentiment..."):
                # Predict sentiment using the selected model and Word2Vec model
                sentiment, confidence, probabilities = predict_sentiment(user_review, selected_model, w2v_model)
                if sentiment:
                    # Results display
                    col1, col2 = st.columns(2)
                    with col1:
                        # Sentiment result
                        emoji = "😊" if sentiment == "Positive" else "😞"
                        color = "success" if sentiment == "Positive" else "error"
                        if sentiment == "Positive":
                            st.success(f"{emoji} **{sentiment}** Sentiment")
                        else:
                            st.error(f"{emoji} **{sentiment}** Sentiment")
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col2:
                        # Probability breakdown
                        st.subheader("📊 Probability Breakdown")
                        prob_df = pd.DataFrame({
                            "Sentiment": ["Negative", "Positive"],
                            "Probability": probabilities
                        })
                        fig, ax = plt.subplots(figsize=(6, 4))
                        colors = ['#e74c3c', '#2ecc71']
                        bars = ax.bar(prob_df["Sentiment"], prob_df["Probability"], color=colors, alpha=0.7)
                        ax.set_ylabel("Probability")
                        ax.set_title("Sentiment Probabilities")
                        ax.set_ylim(0, 1)
                        # Add value labels on bars
                        for bar, prob in zip(bars, probabilities):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{prob:.2f}', ha='center',
                                    va='bottom')
                        st.pyplot(fig)

                    # Additional insights
                    if confidence > 0.8:
                        st.info("🎯 High confidence prediction - the model is very sure about this sentiment.")
                    elif confidence > 0.6:
                        st.info("👍 Moderate confidence prediction - the model is reasonably sure about this sentiment.")
                    else:
                        st.warning("⚠️ Low confidence prediction - the sentiment might be ambiguous or neutral.")
                else:
                    st.warning("⚠️ Unable to analyze the review. Please try a different review.")
        else:
            st.warning("⚠️ Please enter a movie review to analyze.")

    # Example reviews
    st.subheader("💡 Try These Examples")
    examples = [
        ("Positive",
         "This movie was absolutely incredible! The storyline was captivating, the acting was phenomenal, and the special effects were mind-blowing. I couldn't take my eyes off the screen for a single moment."),
        ("Negative",
         "This was one of the worst movies I've ever seen. The plot made no sense, the acting was terrible, and it felt like a complete waste of time. I want my money back."),
        ("Mixed",
         "The movie had some good moments with decent special effects, but the story was confusing and the pacing was off. It's an okay watch if you have nothing else to do.")
    ]
    for label, example in examples:
        if st.button(f"Try {label} Example", key=f"example_{label}"):
            st.text_area("Example review:", value=example, height=100, key=f"display_{label}")


# Function to display information about the project
def show_about_page():
    """
    Display information about the project.
    """
    # Display the header
    st.header("ℹ️ About This Project")

    # Project description
    st.markdown("""
           ## 🎬 Movie Review Sentiment Analysis

    This application demonstrates a complete end-to-end sentiment analysis pipeline for movie reviews using machine learning and natural language processing techniques.

    ### 🔧 Technical Implementation

    Data Processing Pipeline:
    - Data Loading: IMDb Large Movie Review Dataset with 50,000 reviews
    - Text Preprocessing: HTML removal, tokenization, stopword removal, lemmatization
    - Feature Engineering: Word2Vec embeddings for numerical representation
    - Model Training: Logistic Regression and Random Forest classifiers
    - Evaluation: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC

    Key Features:
    - 🔄 Modular Design: Clean separation of concerns with dedicated modules
    - ⚡ Performance Optimization: Streamlit caching for faster loading
    - 📊 Interactive Visualizations: Real-time charts and confusion matrices
    - 🎯 Real-time Predictions: Instant sentiment analysis for user input
    - 📈 Model Comparison: Side-by-side evaluation of different algorithms

    ### 🛠️ Technology Stack
    
    - Python: Core programming language
    - Streamlit: Interactive web application framework
    - Scikit-learn: Machine learning models and evaluation
    - NLTK: Natural language processing
    - Gensim: Word2Vec embeddings
    - Pandas & NumPy: Data manipulation and numerical operations
    - Matplotlib & Seaborn: Data visualization
    

    ### 📊 Model Performance

    The trained models achieve excellent performance on the test dataset:
    - Logistic Regression: ~85% accuracy, 0.925 ROC-AUC
    - Random Forest: ~83% accuracy, 0.908 ROC-AUC

    
    ### 👨‍💻 Group Members

    This project was completed by:
    - Selasi Yohanes Dovi : 22253151
    - Sandra Animwaa Bamfo : 22256394
    - Sharifatu Musah : 22255054
    - Samuel Kwadwo Osae Boateng : 22253881

    ---

    Developed by Group 2 using Python and Streamlit
    """)

if __name__ == "__main__":
    main()

