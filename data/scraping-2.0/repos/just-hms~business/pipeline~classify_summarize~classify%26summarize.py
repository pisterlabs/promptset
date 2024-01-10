# ----------------------------------------------------- JSON TO CSV CONVERTER -----------------------------------------------------
import csv
import json

# Read the JSON file
with open('news.json', 'r', encoding='utf-8') as json_file:
    data = json_file.readlines()

# Extract the header from the first line of JSON
header = json.loads(data[0]).keys()

# Specify the output CSV file path
csv_file = 'output.csv'

# Write the data to the CSV file with UTF-8 encoding
with open(csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)  # Write the header

    for line in data:
        item = json.loads(line)
        writer.writerow(item.values())
# ----------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- CONCATS TITLE TO ARTICLE IN THE CSV ------------------------------------------------
input_file = 'output.csv'
output_file = 'news.csv'

with open(input_file, 'r', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    rows = [row for row in reader]

merged_rows = [[row[1] + ". " + row[4]] + [row[0], row[2], row[3]] for row in rows]


with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerows(merged_rows)
# ----------------------------------------------------------------------------------------------------------------------------------
# ------------------------- CODE THAT PERFORMS CLASSIFICATION ON THE NEWS AND SAVES THEM INTO A NEW CSV ----------------------------
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Load the saved model from the file
with open('sgd_classifier_model.pkl', 'rb') as file:
    sgd_classifier = pickle.load(file)

# Load the new dataset from the CSV file
df = pd.read_csv('news.csv')

# Change the name of the column
df.rename(columns={'title. content': 'article'}, inplace=True)
df.rename(columns={'data': 'date'}, inplace=True)

# Save the DataFrame back to a CSV file
df.to_csv('news.csv', index=False)
df = pd.read_csv('news.csv')

# Function that performs text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

# Apply text cleaning to the new dataset
df['article'] = df['article'].map(lambda article: clean_text(article))

# Load the vectorizer used during training
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Transform the new dataset into vectors using the same vectorizer
new_vectors = vectorizer.transform(df['article'])

# Predict the categories for the new dataset
predicted_categories = sgd_classifier.predict(new_vectors)

# Add the predicted categories as a new column in the original dataset
df_new = pd.read_csv("news.csv")
df_new['category'] = predicted_categories

# Save the updated dataset to the original CSV file
df_new.to_csv('output.csv', index=False)

print("Classification completed successfully and saved as output.csv!")
# ----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------- SUMMARIZE ARTICLES WITH OPENAI API AND STORES IT IN A NEW CSV ------------------------------------
import time
import openai
import csv
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")


# Function to summarize the article using ChatGPT
def summarize_article(article):
    # Generate a prompt for ChatGPT
    prompt = f"Summarize the article shortly:\n\n{article}"

    # Make a request to ChatGPT
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',  # Choose an appropriate engine
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract the generated summary from the response
    summary = response.choices[0].message.content

    return summary


# Open the CSV file for reading and writing
with open('output.csv', 'r', encoding='utf-8', newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames + ['summary']  # Add 'summary' as a new column

    # Open a new CSV file for writing with the added 'summary' column
    with open('news.csv', 'w', encoding='utf-8', newline='') as new_csv_file:
        writer = csv.DictWriter(new_csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each row in the original CSV file
        for row in reader:
            article = row['article']
            category = row['category']
            url = row['url']
            urlToImage = row['urlToImage']
            date = row['date']

            # Summarize the article using the ChatGPT API
            summary = summarize_article(article)
            time.sleep(18)

            # Write the row with the added 'summary' column to the new CSV file
            row['summary'] = summary
            writer.writerow(row)
# ----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ FINALLY CONVERT THE CSV TO JSON FOR HMS -----------------------------------------------
import csv
import json

csv_file = 'news.csv'  # Replace with your CSV file path
json_file = 'news_final.json'  # Replace with the desired JSON file path

with open(csv_file, 'r', encoding='utf-8') as file:
    csv_data = csv.DictReader(file)
    json_data = [row for row in csv_data]

with open(json_file, 'w') as file:
    json.dump(json_data, file, indent=4)
# ----------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- DELETE USELESS FILES ----------------------------------------------------------
import os

csv_file = 'news.csv'  # Replace with the name of your CSV file

if os.path.exists(csv_file):
    os.remove(csv_file)

csv_file = 'output.csv'  # Replace with the name of your CSV file

if os.path.exists(csv_file):
    os.remove(csv_file)
