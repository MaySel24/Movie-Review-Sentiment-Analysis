# Import necessary libraries
import re  # For regular expressions
import string  # For string operations
from nltk.corpus import stopwords  # For stopwords
from nltk.stem import WordNetLemmatizer  # For lemmatization
from nltk.tokenize import word_tokenize  # For tokenization
import nltk  # For natural language processing
import logging  # For logging events and errors

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO
logger = logging.getLogger(__name__)  # Get the logger instance

# Download NLTK resources if they don't exist
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Downloading NLTK wordnet...")
    nltk.download("wordnet")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    logger.info("Downloading NLTK punkt...")
    nltk.download("punkt")

'''try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("Downloading NLTK punkt_tab...")
    nltk.download("punkt_tab")
'''

# Define stopwords and lemmatizer
STOPWORDS = set(stopwords.words("english"))  # Get English stopwords
LEMMATIZER = WordNetLemmatizer()  # Create a lemmatizer instance

# Function to clean text
def clean_text(text):
    """
    Cleans the input text by removing HTML tags, punctuation, and non-alphabetic characters. Converts text to lowercase.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)

    # Remove non-alphabetic characters
    text = re.sub(r"\W", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Function to tokenize text
def tokenize_text(text):
    """
    Tokenizes the input text into words.

    Args:
        text (str): The input string to tokenize.

    Returns:
        list: A list of tokens.
    """
    # Tokenize text using NLTK's word_tokenize function
    return word_tokenize(text)

# Function to remove stopwords
def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.

    Args:
        tokens (list): A list of word tokens.

    Returns:
        list: A list of tokens with stopwords removed.
    """
    # Use list comprehension to filter out stopwords
    return [word for word in tokens if word not in STOPWORDS]

# Function to lemmatize text
def lemmatize_text(tokens):
    """
    Lemmatizes a list of tokens.

    Args:
        tokens (list): A list of word tokens.

    Returns:
        list: A list of lemmatized tokens.
    """
    # Use list comprehension to lemmatize tokens
    return [LEMMATIZER.lemmatize(word) for word in tokens]

# Function to preprocess text
def preprocess_text(text):
    """
    Applies a full preprocessing pipeline to the input text.

    Args:
        text (str): The input string to preprocess.

    Returns:
        list: A list of preprocessed and lemmatized tokens.
    """
    # Clean text
    text = clean_text(text)

    # Tokenize text
    tokens = tokenize_text(text)

    # Remove stopwords
    tokens = remove_stopwords(tokens)

    # Lemmatize tokens
    tokens = lemmatize_text(tokens)

    return tokens

if __name__ == '__main__':
    # Example usage and testing of the text preprocessing functions
    sample_text = "<p>This is an amazing movie! It's so good, I loved it. Don't miss it.</p>"
    logger.info(f"Original text: {sample_text}")

    # Clean text
    cleaned_text = clean_text(sample_text)
    logger.info(f"Cleaned text: {cleaned_text}")

    # Tokenize text
    tokens = tokenize_text(cleaned_text)
    logger.info(f"Tokens: {tokens}")

    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    logger.info(f"Tokens after stopword removal: {filtered_tokens}")

    # Lemmatize tokens
    lemmatized_tokens = lemmatize_text(filtered_tokens)
    logger.info(f"Lemmatized tokens: {lemmatized_tokens}")

    if __name__ == '__main__':
        # Example usage and testing of the text preprocessing functions
        sample_text = "<p>This is an amazing movie! It's so good, I loved it. Don't miss it.</p>"
        logger.info(f"Original text: {sample_text}")

        # Clean text
        cleaned_text = clean_text(sample_text)
        logger.info(f"Cleaned text: {cleaned_text}")

        # Tokenize text
        tokens = tokenize_text(cleaned_text)
        logger.info(f"Tokens: {tokens}")

        # Remove stopwords
        filtered_tokens = remove_stopwords(tokens)
        logger.info(f"Tokens after stopword removal: {filtered_tokens}")

        # Lemmatize tokens
        lemmatized_tokens = lemmatize_text(filtered_tokens)
        logger.info(f"Lemmatized tokens: {lemmatized_tokens}")

        # Preprocess text
        final_tokens = preprocess_text(sample_text)
        logger.info(f"Final preprocessed tokens: {final_tokens}")

        # Test with a more complex example
        complex_text = "The film, while visually stunning, lacked a coherent plot and strong character development. It was a disappointment!"
        logger.info(f"\nOriginal complex text: {complex_text}")

        # Preprocess complex text
        final_complex_tokens = preprocess_text(complex_text)
        logger.info(f"Final preprocessed complex tokens: {final_complex_tokens}")
