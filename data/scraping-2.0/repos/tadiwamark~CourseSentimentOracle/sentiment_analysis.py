import openai
import pickle
import streamlit as st
import tensorflow as tf
from text_preprocessor import preprocess_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from collections import Counter
import urllib.request
import tempfile
import os


# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load pre-fitted tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def load_uploaded_model(model_file, tokenizer_file):
    """Loads a model and tokenizer uploaded by the user."""
    # Save uploaded model to a temporary file
    temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    with open(temp_model_file.name, 'wb') as f:
        f.write(model_file.getbuffer())
    
    # Load the model
    model = tf.keras.models.load_model(temp_model_file.name)

    # Clean up temporary file
    os.remove(temp_model_file.name)

    # Load the tokenizer
    tokenizer = pickle.loads(tokenizer_file.getvalue())
    
    return model, tokenizer
    
# Load custom model
def load_model_from_github(url):
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    loaded_model = tf.keras.models.load_model(filename)
    return loaded_model

# Download and Load Model
model_url = 'https://github.com/tadiwamark/CourseSentimentOracle/releases/download/sentiment-analysis/final_model.h5'
custom_model = load_model_from_github(model_url)

# Constants
MAX_LEN = 100
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"


def analyze_sentiment_simple(review_text,tokenizer=None,model=None):
    # Preprocess the review text
    preprocessed_text = preprocess_text(review_text)

    # Performing NLP using spaCy
    doc = nlp(preprocessed_text)

    # Extracting entities
    entities = [ent.text for ent in doc.ents]

    # Extracting keywords using noun chunks
    keywords = [chunk.text for chunk in doc.noun_chunks]

    # Tokenize the preprocessed text
    # If tokenizer is not provided, use the default one
    if tokenizer is None:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])

    # Use the default custom_model if no model is passed
    if model is None:
        model = custom_model

    # Check if tokenizer returned an empty list
    if not sequence or not sequence[0]:
        return "Review contains words not seen during training. Unable to process."

    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    # Predict the sentiment
    prediction = custom_model.predict(padded_sequence)

     # Convert prediction to sentiment label
    sentiment_result = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    
    return sentiment_result, keywords, entities


def advanced_sentiment_analysis(review_text, model='gpt-3.5-turbo'):
    """
    Perform advanced sentiment analysis based on the selected model.

    Parameters:
    - review_text (str): The review text to analyze.
    - model (str): The model to use for analysis, either 'gpt-3.5-turbo' or 'simple'.

    Returns:
    - sentiment_result (str): The resulting sentiment.
    - additional_features (dict): Any additional features extracted, like entities or keywords.
    """

    if model == 'gpt-3.5-turbo':
        preprocessed_text = preprocess_text(review_text)
        if openai.api_key:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant. Provide sentiment, keywords, and entities separately."},
                {"role": "user", "content": f"For the review: '{preprocessed_text}', what is its sentiment? Also, list its keywords and entities."}
            ]
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            temperature=0.5,
            max_tokens=150
            )
        
            response_content = response['choices'][0]['message']['content'].split('\n')
        

            sentiment_result = response_content[0].replace("Sentiment:", "").strip()
            entities = response_content[1].replace("Entities:", "").split(',')
            keywords = response_content[2].replace("Keywords:", "").split(',')
        
            
        

    elif model == 'simple':
        # For the simple model, you'd process the review differently.
        sentiment_result ,additional_features = analyze_sentiment_simple(review_text)
        
    else:
        raise ValueError(f"Model {model} not supported.")

    return sentiment_result, keywords, entities


def additional_nlp_features(keywords,entities):
    """Display additional NLP features."""
    if keywords:
        st.subheader("Extracted Keywords:")
        st.write(keywords)

        st.subheader("Recognized Entities:")
        st.write(entities)
