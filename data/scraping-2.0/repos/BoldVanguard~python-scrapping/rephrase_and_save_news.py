import os
import json
import glob
import spacy
import firebase_admin

from firebase_admin import credentials
from firebase_admin import db
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date
from openai import OpenAI
from dotenv import load_dotenv

# gets API Key from environment variable OPENAI_API_KEY
load_dotenv()

# Load the medium-sized English model of spaCy
nlp = spacy.load("en_core_web_md")

# Function to preprocess text using spaCy
def preprocess(text):
    doc = nlp(text)
    processed_text = " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
    return processed_text

# Initialize Firebase app with credentials
cred = credentials.Certificate(os.environ.get("CRED"))
firebase_admin.initialize_app(cred, {
    'databaseURL': os.environ.get("DATABASE_URL")
})


def push_data_to_firebase(data):
    try:
        # Get a reference to the Firebase Realtime Database root
        ref = db.reference('articles')

        # Push data to Firebase RTDB
        new_article_ref = ref.push(data)
        print(f"Data pushed successfully to Firebase RTDB with key: {new_article_ref.key}")

    except Exception as e:
        print(f"Failed to push data to Firebase RTDB: {e}")

# Function to calculate cosine similarity between articles
def calculate_similarity(text1, text2):
    processed_text1 = preprocess(text1)
    processed_text2 = preprocess(text2)
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([processed_text1, processed_text2])
    
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim

# Function to find similar articles in a database
def when_add_new_article(database, new_article):
    similarity_threshold = 0.5  # Set a threshold for similarity
    
    for i in range(len(database)):
        
        with open(database[i], 'r', encoding='utf-8') as file:
            # Load the JSON data
            data = json.load(file)
        articlei = data["content"]
        similarity = calculate_similarity(articlei, new_article)
        if similarity > similarity_threshold:
            return 1
    return 0

client = OpenAI()

def setup_directory(base_dir, sub_dir):
    path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

base_path = "converted_articles"
sub_folder = date.today().strftime('%Y-%m-%d')
raw_directory = setup_directory("raw_articles", sub_folder)

# Construct the pattern to match JSON files in the specified directory
json_pattern = os.path.join(raw_directory, '*.json')

# Use glob to find all JSON files in the directory
json_files = glob.glob(json_pattern)
json_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

json_pattern = os.path.join(base_path, '*.json')


print("----- standard request -----")
for json_file in json_files:
    
    json_articles = glob.glob(json_pattern)
    # Open the JSON file
    with open(json_file, 'r', encoding='utf-8') as file:
        # Load the JSON data
        data = json.load(file)
    
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "How to convert this sentence in same meaning with under 36 characters but different, I just want converted sentence, no need for any description" + data["title"]},
        ],
    )

    rewritten_title = completions.choices[0].message.content

    completions = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "How to convert this paragraph in same meaning under 100 lines but different, I just want converted paragraph, no need for any description" + data["content"]},
        ],
    )

    rewritten_content = completions.choices[0].message.content
    
    if(when_add_new_article(json_articles, rewritten_content) == 0):
        json_file_path = str(len(json_articles) + 1) + '.json'
        destination_blob_name = "article/" + json_file_path
        print("New article is added in " + json_file_path)
        file_path = os.path.join(base_path, json_file_path)
        
        data_to_push = {
            "author": data['author'],
            "content": rewritten_content,
            'date':data['date'],
            'imageUrl':data['imageUrl'],
            'title': rewritten_title
        }

        with open(file_path, "w") as file:
            json.dump(data_to_push, file, indent = 4)

        # Push data to Firebase RTDB
        push_data_to_firebase(data_to_push)
    else:
        print("Similar article is detected, not saved! File_path:" + json_file)