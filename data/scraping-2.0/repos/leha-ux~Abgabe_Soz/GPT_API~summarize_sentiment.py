
import jsonlines                    # zum Laden der Daten
import openai
from dotenv import load_dotenv
import os
import json
import time
import nltk



from collections import Counter     # um worte zu zählen
import matplotlib.pyplot as plt     # Für Visualisierung

load_dotenv()


# Authentication
openai.api_key = os.getenv("OPENAI_KEY", "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")



#------------------------------------------------------------------

def read_json_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data



################# SPEICHERN DER DATEN ####################
def save_dict_to_json(dictionary, speech_id):
    # Generate the file name with the idea
    file_name = f"../../data/speeches_sentiment/{speech_id}_data.json"
    # Open the file in write mode and save the dictionary as JSON
    with open(file_name, 'w') as json_file:
        json.dump(dictionary, json_file)



################# SPEICHERN DER DATEN ####################

# Hier legen wir fest, welche Daten (Wahlperiode 19 oder 20) wir laden:
legislatur = 20

# Wir generieren eine leere Liste:
alleReden = []

# Wir öffnen den entsprechende File (Dateipfad anpassen!):
with jsonlines.open(f'../../data/speeches_{legislatur}.jsonl') as f:
    for line in f.iter():
        # Wir packen alles Zeile für Zeile zu unserer Liste:
        alleReden.append(line)

# Wir sortieren nach Datum:
alleReden.sort(key = lambda x :x['date'])

################# FUNKTIONEN #####################

timeout_seconds = 5
retry_delay_seconds = 1

import time

import os
import json

# File to store processed speech ids
processed_speeches_file = "processed_speeches_sentiment.txt"

def call_api_with_retry(sentence):
    retry = True
    while retry:
        try:
            speech_string = f'''Hier ist ein Satz einer politischen Rede: {sentence}
            Bewerte das Sentiment (neutral, positiv, negativ).
            "Sentiment:"'''
            chat_history = [{"role": "user", "content": speech_string}]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-16k-0613',
                messages=chat_history
            )
            message = response.choices[0]['message']
            print("{}: {}".format(message['role'], message['content']))

            # If response received successfully, set retry to False to break the loop
            retry = False
            
            return message['content']
        except Exception as e:
            # Check if error is due to server overload or not ready
            if 'overloaded' in str(e) or 'not ready' in str(e):
                print("Server is overloaded or not ready yet. Retrying after 1 second.")
                time.sleep(retry_delay_seconds)
            else:
                print(f"An error occurred: {e}")
                # If error is due to other reasons, raise the exception and stop the execution
                raise



nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Get already processed speeches
if os.path.exists(processed_speeches_file):
    with open(processed_speeches_file, 'r') as f:
        processed_speeches = f.read().splitlines()
else:
    processed_speeches = []

for speech in alleReden:
    # Only process the speech if it hasn't been processed yet
    if speech["id"] not in processed_speeches:
        sentences = sent_tokenize(speech["text"])
        speech_sentiments = {"id": speech["id"], "sentiments": []}
        for sentence in sentences: 
            sentiment = call_api_with_retry(sentence)
            speech_sentiments["sentiments"].append({"sentence": sentence, "sentiment": sentiment})
        # Mark this speech as processed by adding its id to the processed_speeches_file
        with open(processed_speeches_file, 'a') as f:
            f.write(speech["id"] + "\n")
        save_dict_to_json(speech_sentiments, speech["id"])




