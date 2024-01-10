# # Code inspired from article https://towardsdatascience.com/text-summarization-with-nlp-textrank-vs-seq2seq-vs-bart-474943efeb09  

## for data
import datasets 
import pandas as pd 
import numpy as np

## for preprocessing
import re
import nltk 
#import contractions 
import openai
## for textrank
import pytextrank
import spacy
#for LSA summarize
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
##for generating sentence after textrank 
#import openai
import torch
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import pprint as pprint


# Define a function to apply textrank algorithm to corpus with a ratio parameter
def ptrank(nlp, corpus, ratio=0.2):
    if type(corpus) is str:
        corpus = [corpus]
    lst_phrases = []
    for txt in corpus:
        # Parse the document with spaCy
        doc = nlp(txt)
        # Extract the top-ranked phrases from the document
        phrases = []
        for phrase in doc._.phrases:
            phrases.append((phrase.text, phrase.rank))
        # Sort the phrases by rank
        phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        # Limit the number of phrases based on the ratio
        limit = int(len(phrases) * ratio)
        # Append only the top phrases to the lst_phrases list
        lst_phrases.append(phrases[:limit])
    return lst_phrases


def train_sum():
  # import dataset 
  ## load the full dataset of 300k articles
  dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')
  lst_dics = [dic for dic in dataset["train"]]

  ## keep the first N articles if you want to keep it lite 
  dtf = pd.DataFrame(lst_dics).rename(columns={"article":"text", 
        "highlights":"y"})[["text","y"]].head(20000)
  dtf.head()

  #check one example 
  i = 0
  print("--- Full text ---")
  print(dtf["text"][i])
  print("--- Summary ---")
  print(dtf["y"][i])

  # # create train/test dataset
  dtf_train = dtf.iloc[i+1:]
  dtf_test = dtf.iloc[:i+1]

  # # TextRank algorithm

  # Load a spaCy model
  nlp = spacy.load("en_core_web_sm")

  # Add PyTextRank to the spaCy pipeline
  nlp.add_pipe("textrank")

  # Apply the function to corpus with a ratio of 0.2
  predicted  = ptrank(nlp, corpus=dtf_test["text"], ratio=0.2)
  predicted [i]

  # ## Result using basic concatenation of top[ratio] words 
  dirty_sentence = " ".join([str(t[0]) for t in predicted[i]])
  print(dirty_sentence)

# ## Generate content using GPT-2 
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

import numpy as np

generator(dirty_sentence, max_length=100, num_return_sequences=5)

# ## Result using keyword and OPEN API GPT3 Davinci (billing problem for now, contacted openAI)
with open("open_ai_api_key.txt", "r") as f:
    openai.api_key = f.read().rstrip()

separator = ","

# Format predicted as a partial text
input = "Generate a sentence from the following keywords:\n\n"
for keyword, value in predicted[i]:
    input += f"- {keyword}, {value}\n"
input += "\nSentence:"

# Send a request to the OpenAI API using the text completion feature
response = openai.Completion.create(
    engine="davinci",
    prompt=input,
    max_tokens=50,
    temperature=0.5,
    frequency_penalty=0.5,
)

# Print the generated sentence
print(response["choices"][0]["text"])

# # Using LSA Method 
parser = PlaintextParser.from_string(dtf_train["text"], Tokenizer("english"))
summarizer_lsa = LsaSummarizer()
summary = summarizer_lsa(parser.document, 8)
for sentence in summary:
    pprint.pprint(sentence)