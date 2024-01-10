from sqlalchemy import create_engine
import sqlite3
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Connect to the SQLite database
conn = sqlite3.connect('data/stocks.db')

# Add a sentiment column to the stock_news table if it does not already exist
conn.execute("ALTER TABLE stock_news ADD COLUMN sentiment INTEGER")

# Define a query to get the news headlines
query = """
SELECT datetime, Stock, summary 
FROM stock_news;
"""

# Read the data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

# Define a function to get the sentiment of a text
def get_sentiment(text):
    if not text:
        return None
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()
    return sentiment

# Get the sentiment for each news headline
df['sentiment'] = df['summary'].apply(get_sentiment)

# Update the stock_news table with the sentiment scores
for index, row in df.iterrows():
    conn.execute("UPDATE stock_news SET sentiment = ? WHERE datetime = ? AND Stock = ?", (row['sentiment'], row['datetime'], row['Stock']))

# Commit the changes and close the connection
conn.commit()
conn.close()

print('Sentiment analysis completed and updated in the stock_news table!')







# from sqlalchemy import create_engine
# import sqlite3
# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification
# from torch.nn.functional import softmax
# import torch
#
# # Load pre-trained model and tokenizer
# model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
#
# # Connect to the SQLite database
# conn = sqlite3.connect('data/stocks.db')
#
# # Define a query to get the news headlines
# query = """
# SELECT datetime, Stock, summary
# FROM stock_news
# WHERE datetime BETWEEN '2023-08-01' AND '2023-08-31';
# """
#
# # Read the data into a pandas DataFrame
# df = pd.read_sql_query(query, conn)
#
# # Close the connection
# conn.close()
#
# # Define a function to get the sentiment of a text
# def get_sentiment(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     outputs = model(**inputs)
#     probs = softmax(outputs.logits, dim=-1)
#     sentiment = torch.argmax(probs, dim=-1).item()
#     return sentiment
#
# # Get the sentiment for each news headline
# df['sentiment'] = df['headline'].apply(get_sentiment)
#
# # Store the results in a new SQLite table
# engine = create_engine('sqlite:///data/stocks.db')
# df.to_sql('stock_news_sentiment', engine, index=False, if_exists='replace')

# print('Sentiment analysis completed and stored in the SQLite database!')



# import openai
# import sqlite3
# import pandas as pd
# from sqlalchemy import create_engine
# # OpenAI API key
# openai.api_key = "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"
#
# # Connect to the SQLite database
# conn = sqlite3.connect('data/stocks.db')
#
# # Define a query to get the news headlines
# query = """
# SELECT datetime, Stock, headline
# FROM stock_news
# WHERE datetime BETWEEN '2023-08-01' AND '2023-08-31'
# AND Stock = "AAPL";
# """
#
# # Read the data into a pandas DataFrame
# df = pd.read_sql_query(query, conn)
#
# # Close the connection
# conn.close()
#
# # Define a function to get the sentiment of a text
# def get_sentiment(text):
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=f"The sentiment of the following text is: {text}",
#         max_tokens=5,
#     )
#     sentiment = response.choices[0].text.strip()
#     return sentiment
#
# # Get the sentiment for each news headline
# df['sentiment'] = df['headline'].apply(get_sentiment)
#
# # Store the results in a new SQLite table
# engine = create_engine('sqlite:///data/stocks.db')
# df.to_sql('stock_news_sentiment', engine, index=False, if_exists='replace')
#
# print('Sentiment analysis completed and stored in the SQLite database!')
