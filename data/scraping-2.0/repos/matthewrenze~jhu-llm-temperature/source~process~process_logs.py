# Import the packages
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from Levenshtein import distance as get_distance
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import tiktoken
import openai

# Set the parameters
input_folder = f"../../logs"
output_folder = f"../../logs"

def get_jaccard(a, b):

    # Tokenize the strings into sets of words
    set1 = set(a)
    set2 = set(b)

    # Calculate the Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    similarity = intersection / union if union != 0 else 0.0

    return similarity

def get_jaccard_simiarity(texts):

    # Get all combinations of texts as tuples
    combos = list(combinations(texts, 2))

    # Calculate the Jaccard similarity for each tuple
    similarities = [get_jaccard(a, b) for a, b in combos]

    # Return the mean similarity
    return np.mean(similarities)

def get_bag_of_words_similarity(texts):

    # Create the vectorizer
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(texts)

    # Get the cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    # Get the upper triangle
    upper_matrix = np.triu_indices_from(similarity_matrix, k=1)

    # Get the average similarity
    avg_similarity = np.mean(similarity_matrix[upper_matrix])

    return avg_similarity

def get_tfidf_similarity(texts):

    # Create the vectorizer
    vectorizer = TfidfVectorizer()

    # Create the vectors
    vectors = vectorizer.fit_transform(texts)

    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(vectors)

    upper_matrix = np.triu_indices_from(similarity_matrix, k=1)

    avg_similarity = np.mean(similarity_matrix[upper_matrix])

    return avg_similarity

def get_levenshtein_distance(texts):

    distances = [get_distance(a, b) for a, b in combinations(texts, 2)]

    avg_distance = np.mean(distances)

    return avg_distance

def get_bleu_score(texts):

    # Calculate the average BLEU score
    bleu_scores = [sentence_bleu([a], b) for a, b in combinations(texts, 2)]

    avg_bleu_score = np.mean(bleu_scores)

    return avg_bleu_score

def get_sbert_similarity(texts):

        # Create the model
        # model = SentenceTransformer("all-mpnet-base-v2")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Get the embeddings
        embeddings = model.encode(texts)

        # Calculate the cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Get the upper triangle
        upper_matrix = np.triu_indices_from(similarity_matrix, k=1)

        # Get the average similarity
        avg_similarity = np.mean(similarity_matrix[upper_matrix])

        return avg_similarity

def get_char_length(texts):

    # Get the character count
    char_count = [len(text) for text in texts]

    # Get the average character count
    avg_char_length = np.mean(char_count)

    return avg_char_length

def get_word_length(texts):

    # Get the word count
    word_count = [len(text.split()) for text in texts]

    # Get the average word count
    avg_word_length = np.mean(word_count)

    return avg_word_length

def get_token_length(texts):

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    token_count = [len(encoding.encode(text)) for text in texts]

    avg_token_count = np.mean(token_count)

    return avg_token_count



# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Create a table for the logs
logs = pd.DataFrame()

# Get the files recursively
file_names = []
for root, dirs, files in os.walk(input_folder):

    # Skip the "_archive" folder
    if "_archive" in root:
        continue

    # Loop through each file
    for file in files:
        file_name = f"{root}/{file}"
        file_name = file_name.replace("\\", "/")
        file_names.append(file_name)

# Include only csv files with 100 questions
file_names = [file for file in file_names if file.endswith("-100.txt")]

# DEBUG: Include only the first 11 files
# file_names = file_names[:11]

# Loop through each file
for file_name in file_names:

    # Display a status update
    print(f"Processing {file_name}")

    # Get the agent from the file path
    agent_name = re.search(r"logs/(.*)/", file_name).group(1)
    agent_name = agent_name.replace("_", "-").title()
    agent_name = agent_name.replace("-Of-", "-of-")

    # Get the model from the file name
    model_name = re.search(r"temp-\d\.\d - (.*) - (.*)\.txt", file_name).group(1)

    # Rename GPT-3.5
    if model_name == "gpt-35-turbo":
        model_name = "gpt-3.5"

    # Get the temperature from the file name
    temperature = float(re.search(r"temp-(\d\.\d)", file_name).group(1))

    # Add the additional properties to the table
    log = {}
    log["Agent"] = agent_name
    log["Model"] = model_name
    log["Temperature"] = temperature

    # Remove file extension from exam
    log["Exam"] = re.search(r"temp-\d\.\d - (.*) - (.*)\.txt", file_name).group(2)

    # Read the file as text
    with open(file_name, "r", encoding="utf-8") as file:
        content = file.read()

    # Replace multiple newlines with single newlines
    content = re.sub(r"\n{2,}", "\n", content)

    # Use regular expression to match each problem block
    pattern = r"(^### .+? ###\n(.+?\n)+?)(?=^### |$)"
    problems = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    problems = [problem[0] for problem in problems]

    # Skip if there are no problems
    if len(problems) == 0:
        continue

    # Loop through each problem
    for i, problem in enumerate(problems):

        # Display a status update
        print(f" - Processing problem {i + 1} of {len(problems)}")

        # Skip results
        if problem.startswith("### Results ###"):
            continue

        # Use regular expression to match each response body
        pattern = r"Response \d+:\n(.+?)(?=Response \d+:|Votes:|$)"
        responses = re.findall(pattern, problem, re.DOTALL)

        # Skip if there are no responses
        if len(responses) == 0:
            continue

        # Add the problem number
        log["Problem"] = i + 1

        # Get the similarity scores
        log["Jaccard Similarity"] = get_jaccard_simiarity(responses)
        log["BoW Similarity"] = get_bag_of_words_similarity(responses)
        log["TF-IDF Similarity"] = get_tfidf_similarity(responses)
        log["Levenshtein Distance"] = get_levenshtein_distance(responses)
        log["BLEU Score"] = get_bleu_score(responses)
        log["SBERT Similarity"] = get_sbert_similarity(responses)
        log["Character Length"] = get_char_length(responses)
        log["Word Length"] = get_word_length(responses)
        log["Token Length"] = get_token_length(responses)

        # Add the log to the logs table
        logs = logs._append(log, ignore_index=True)

# Save the logs
logs.to_csv(f"{output_folder}/logs.csv", index=False)
