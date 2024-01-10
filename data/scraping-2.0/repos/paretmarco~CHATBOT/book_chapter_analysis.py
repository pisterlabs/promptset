import json
import os
import numpy as np
import openai
from config import OPENAI_API_KEY
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google_sheets_helper import read_data_from_sheet, find_sheet_by_title, write_data_to_sheet

openai.api_key = OPENAI_API_KEY
MODEL = "gpt-3.5-turbo"

with open('config.json') as config_file:
    config = json.load(config_file)
    author_personality = config['author_personality']
    sheet_name = config['sheet_name']  # Read the sheet_name from the config file
    language = config['language']

def calculate_similarities(book_data):
    chapters = [record[2] for record in book_data]  # Change this line to use the new book_data format
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chapters)
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return cosine_similarities

def find_most_similar_chapters(similarity_matrix, index, num_chapters=3):
    similarities = np.array(similarity_matrix[index])
    sorted_indices = np.argsort(similarities)[::-1]
    top_similar_chapters = [idx for idx in sorted_indices[1:num_chapters+1] if idx != index]
    return top_similar_chapters

def generate_suggestions(chapter_title, chapter_text, similar_chapters):
    messages = [
        {"role": "system", "content": f"{author_personality}. You are the writer that completes the chapters in order to make a chapter different from other similar chapters and very unique."},
        {"role": "user", "content": f"Compare the following chapter titled '{chapter_title}':\n{chapter_text}\n with the similar chapters:\n{similar_chapters}\n Speak {language}. Continue and complete it for 100 words inventing suggestions and deepening it putting a perspective that would be very unique and at the same time more related to its title. You can connect to what is already said in the chapter but never repeat it. Insert also a citation."}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            max_tokens=600,
            n=1,
            stop=None,
            temperature=0.7,
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

if __name__ == "__main__":
    # Read the sheet_name from the config file
    with open('config.json') as config_file:
        config = json.load(config_file)
        sheet_name = config['sheet_name']

    # Find the Google Sheet with the specified name
    folder_id = "1O4__jxbTubCObzCCn08oPe0bc0DlP-uO"  # Replace with the folder ID of the "AI_CREATED_BOOKS" directory
    sheet_id = find_sheet_by_title(sheet_name, folder_id)

    # Define the sheet ID and the range for reading data
    READ_RANGE = "A2:C"  # Adjust this range based on your sheet's structure
    sheet_data = read_data_from_sheet(sheet_id, READ_RANGE)

    # Calculate similarity between chapters
    similarity_matrix = calculate_similarities(sheet_data)

    # Analyze all chapters, starting from the last one
    for chapter_index in range(len(sheet_data)-1, -1, -1):
        # Find the most similar chapters
        top_similar_chapters = find_most_similar_chapters(similarity_matrix, chapter_index)

        # Concatenate similar chapters and trim the text to fit within the token limit
        similar_chapters_text = " ".join([sheet_data[i][2] for i in top_similar_chapters])[:1000]

        # Generate suggestions for the chosen chapter
        suggestions = generate_suggestions(sheet_data[chapter_index][1], sheet_data[chapter_index][2], similar_chapters_text)
        print(f"Suggestions for Chapter {sheet_data[chapter_index][1]}:\n{suggestions}\n")

        # Save the updated suggestions in the Google Sheet in column G
        write_data_to_sheet(sheet_id, f"G{chapter_index + 2}", [[suggestions]])
