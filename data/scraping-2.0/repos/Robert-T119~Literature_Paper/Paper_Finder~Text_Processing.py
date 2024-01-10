import re
import openai
import time
import pandas as pd
from .Constants import *
import requests
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import os

# Download NLTK resources
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    return text

def lowercase_text(text):
    return text.lower()

def tokenize_text(text):
    return text.split()

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def is_relevant(abstract):
    if not isinstance(abstract, str):  # Ensure the abstract is a string
        return False
    return any(keyword in abstract.lower() for keyword in must_satisfy_keywords) # Make sure to check in lowercase

def run_prediction(model_name, prompts):
    predictions = []
    for i, prompt in enumerate(prompts):
        prediction = "Error"  # Default prediction in case of errors
        try:
            print(f"Processing prompt {i+1} of {len(prompts)}")
            response = openai.Completion.create(
                model=model_name,
                prompt=prompt,
                max_tokens=1,  # We only need the first token for classification
                stop=stop_sequence
            )
            prediction = response.choices[0].text.strip()
            print(f"Prediction: {prediction}")
        except openai.error.InvalidRequestError:
            print("Prompt too long. Skipping.")
            prediction = "Prompt too long"
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Sleeping for a bit.")
            time.sleep(60)  # Sleep for 60 seconds
        finally:
            predictions.append(prediction)
    return predictions

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {openai.api_key}'
}

def get_embedding(text):
    data = {
        'input': text,
        'model': 'text-embedding-ada-002'
    }
    response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, json=data)
    return np.array(response.json()['data'][0]['embedding'])