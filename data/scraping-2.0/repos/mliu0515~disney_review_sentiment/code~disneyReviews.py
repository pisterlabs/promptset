import psycopg2
from transformers import pipeline
import torch
import warnings
import openai


# Connection parameters
dbname = ...
user = ...
password = ...
host = 'localhost'  # or your server's IP
port = '5432'  # default PostgreSQL port
openai.api_key = ...


# classifier = pipeline("sentiment-analysis")
# TODO: environment variable for API key .env file
# And then we can do something for example os.getenv('BING_API_KEY', '')

# def huggingface_sentiment_analysis(text):
#     # Tokenize the input text and convert to tensor
#     sentiment = classifier(text)
#     return sentiment['label']

def gpt3_sentiment_analysis(text):
    # Query GPT-3
    # TODO: add a try catch block. If fails, wait 1 second, and try again. 
    response = openai.Completion.create(
        engine="davinci",  # Using the davinci engine, but you can choose others.
        prompt=f"Analyze the sentiment of the following text: '{text}'\nSentiment:",
        max_tokens=10
    )

    sentiment = response.choices[0].text.strip()
    return sentiment

# TODO: wrap line 40 - 43 in a function and later on test with pytest.
# TODO: create a migration file for line 40 - 43
# Establish the connection
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

# Create a cursor object
cur = conn.cursor()

# TODO: wrap line 45-52 in a function
# Execute a query
cur.execute("SELECT * FROM reviews")

# Fetch all rows from the result
rows = cur.fetchall()

# put the following two lines in a migration file
cur.execute("ALTER TABLE reviews ADD COLUMN sentiment VARCHAR(225)")
conn.commit()

# TODO: create function to update tables for this for loop block
# __main__: 
for i, row in enumerate(rows):
    row_id = row[0]  # Assuming the first column is an ID or unique identifier
    sentitment = gpt3_sentiment_analysis(row[4])
    cur.execute("UPDATE reviews SET sentiment = %s WHERE review_id = %s", (sentitment, row_id))

conn.commit()

# Close the cursor and the connection
cur.close()
conn.close()
