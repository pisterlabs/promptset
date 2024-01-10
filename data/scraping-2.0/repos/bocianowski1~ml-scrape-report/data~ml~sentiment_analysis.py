from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import openai
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

import re
import pandas as pd
import os

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
nltk_downloads_completed = False

def nltk_downloads():
    global nltk_downloads_completed
    if not nltk_downloads_completed:
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk_downloads_completed = True

def preprocess_text(text: str) -> str:
    try:
        nltk_downloads()
    except:
        pass

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    preprocessed_text = " ".join(words)
    return preprocessed_text

def load_model():
    try:
        model, tokenizer, labels = joblib.load(ABSOLUTE_PATH + 'cached_model.joblib')
        print("Cached model loaded.")
    except FileNotFoundError:
        print("Downloading model...")
        roberta = "cardiffnlp/twitter-roberta-base-sentiment"
        model = AutoModelForSequenceClassification.from_pretrained(roberta, max_length=512)
        tokenizer = AutoTokenizer.from_pretrained(roberta)
        labels = ["Negative", "Neutral", "Positive"]
        
        joblib.dump((model, tokenizer, labels), ABSOLUTE_PATH + 'cached_model.joblib')
        print("Model downloaded and cached.")
    
    return model, tokenizer, labels

def get_sentiment(text, model, tokenizer, labels):
    encoded_text = tokenizer(preprocess_text(text), return_tensors="pt")
    output = model(**encoded_text)

    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    return labels[scores.argmax()], round(scores.max(), 4)

def analyze_dataframe(df, texts, model, tokenizer, labels) -> None:
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = [preprocess_text(text) for text in texts]
    print("Preprocessed texts")

    sentiments = []
    confidences = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        print(f"{i}/{len(texts)}")
        batch_texts = texts[i:i+batch_size]
        encoded_texts = tokenizer.batch_encode_plus(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = model(**encoded_texts)
        logits = outputs.logits
        logits = logits.detach().numpy()

        batch_scores = softmax(logits, axis=1)
        batch_scores = torch.from_numpy(batch_scores)
        batch_preds = torch.argmax(batch_scores, dim=1)

        for j, pred in enumerate(batch_preds):
            print(f"{j}/{len(batch_preds)}", end="\r")
            sentiment = labels[pred]
            confidence = batch_scores[j][pred].item()
            sentiments.append(sentiment)
            confidences.append(confidence)

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences

class Term:
    SHORT = "short"
    LONG = "long"

def create_sentiment_prompt(headline: str, company_name: str, term: str = Term.SHORT) -> str:
    return f"""
        Forget all your previous instructions. Pretend you are a financial expert. 
        You are a financial expert with stock recommendation experience. 
        Answer “YES” if good news, “NO” if bad news, or “UNKNOWN” if uncertain in the first line. 
        Then elaborate with one short and concise sentence on the next line. 
        Is this headline good or bad for the stock price of {company_name} in the {term} term?
        Headline: {headline}
    """

def create_response(prompt: str) -> str:
    if len(prompt) > 2048:
        return "ERROR: Prompt too long. Max length is 2048."
    if len(prompt) < 1:
        return "ERROR: Prompt too short. Min length is 1."
    if os.environ.get("OPENAI_API_KEY") is None:
        return "ERROR: No OpenAI API key found."
    
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    try:
        response = openai.Completion.create(
            model="text-davinci-003", # gpt-3.5-turbo
            prompt=prompt,
            temperature=0,
        )
        return response["choices"][0]["text"]
    except Exception as e:
        return f"ERROR: {e}"
