#import general packages
import pandas as pd
import numpy as np

#import functions
from utils.registry import load_model
from utils.data_preproc import cleaning_text

#import fast_api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#import packages
from transformers import  BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from transformers import BertConfig, AutoTokenizer, TFBertModel, BertTokenizer, TFBertForSequenceClassification, BertModel
import tensorflow as tf
import joblib
import os

import openai
from tensorflow.keras.utils import pad_sequences


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    )


# $WIPE_END



@app.get("/predict_offensive")
def predict_binary(X_pred=str):

    #preprocessing
    X_pred = str(X_pred)
    X_pred = cleaning_text(X_pred)

    #tokenizer_bert
    tokenizer_mini = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
    X_pred = tokenizer_mini(X_pred, return_tensors='tf', padding=True, truncation=True, max_length=512)
    X_pred = dict(X_pred)

    config = BertConfig.from_pretrained('prajjwal1/bert-mini')
    model = TFBertForSequenceClassification(config=config)
    model.build()
    model.load_weights("bert_binary.h5")

    y_pred = model.predict(X_pred)

    probabilities = tf.sigmoid(y_pred[0][0])
    print(probabilities)

    if probabilities[0] < 0.5:
        prediction = "❌ offensive tweet"
    else:
        prediction = "✅ non-offensive tweet"

    print(prediction)

    return {"type of tweet": prediction}


@app.get("/predict_label")
def predict_classif(tweet_a_predire):

    #preprocessing
    data = pd.DataFrame({"text":[tweet_a_predire]})
    data["text"] = data["text"].apply(cleaning_text)


    #tokenizer_bert
    X_pred=data["text"]
    tk = joblib.load("tokenizer.joblib")
    X_pred_tok = tk.texts_to_sequences(X_pred)
    X_pred_pad = pad_sequences(X_pred_tok, dtype='float32', padding='post', maxlen = 180)

    model = load_model("GRU_classif")
    y_pred = model.predict(X_pred_pad)

    probabilities = np.array(y_pred)
    df=pd.DataFrame(probabilities, columns=['Racist tweet','Religious Hate','Xenophobia','Misogyny','Transphobia','Homophobia'])

    label = df.idxmax(axis=1).iloc[0]

<<<<<<< Updated upstream
    print(label)
=======
    #breakpoint()

>>>>>>> Stashed changes
    return {"label": label}



def generate_fight_tweet(tweet, classification):

    openai.api_key = os.environ.get("API_KEY")
    content_of_the_request = f"We have received an offensive tweet. This tweet can be classified as {classification}. Please find here the tweet '{tweet}'. Could you please generate a response to this tweet by explaining that this tweet is {classification} and recall the potential penalties incurred (legally but also in terms of banning on the tweeter platform). Please generate a response in the form of a tweet of max 280 characters and directly generate the quoted response without anything else."
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role':'user','content': content_of_the_request}])
    print(response.choices[0].message.content)
    #print(response)



@app.get("/")
def root():
    return dict(greeting="Hello")


if __name__ == "__main__":
    print(predict_binary(X_pred="Gay are dumb"))
    print(predict_classif("Gay are dumb"))
