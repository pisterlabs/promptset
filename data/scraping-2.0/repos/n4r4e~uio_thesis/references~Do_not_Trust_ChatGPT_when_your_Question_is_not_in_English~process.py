## "Don’t Trust ChatGPT when your Question is not in English: A Study of Multilingual Abilities and Types of LLMs"
## https://aclanthology.org/2023.emnlp-main.491/
## Apply a toy-sized version of the methods from the paper for Korean
# %%
import os
from openai import OpenAI
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
# Load the API key from the .env file
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

#%%
def ask_chatgpt(prompt, model= "gpt-3.5-turbo-1106"):
    """Function to ask ChatGPT, return the response"""
    response = client.chat.completions.create(
        model = model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def translate_to_korean(text, model= "gpt-3.5-turbo-1106"):
    """Function to translate the text to Korean"""
    translation_prompt = f"Translate the following English text to Korean: {text}"
    return ask_chatgpt(translation_prompt, model)

def translate_to_english(text, model="gpt-3.5-turbo-1106"):
    """Function to translate the text to English"""
    translation_prompt = f"Translate the following Korean text to English: {text}"
    return ask_chatgpt(translation_prompt, model)

def process_json_file(file_name, embedding_model):
    # Initialize lists to store the data
    questions = []
    gpt_answers = []
    translated_questions = []
    gpt_answers_translated = []
    translated_back_to_english = []
    cosine_similarities = []
    choices_list = []
    answers = []

    # Read the JSON file and process the data
    with open(file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line)

            # Default values
            question = None
            gpt_answer = None
            translated_question = None
            translated_gpt_answer = None
            translated_back_gpt_answer = None
            cosine_sim = None
            choices = None
            answer = None
            
            if 'Math' in file_name:
                question = json_object.get('question', None)
                answer = json_object.get('answer', None)

            elif 'Logic' in file_name:
                stem = json_object.get('question', {}).get('stem', None)
                choices = json_object.get('question', {}).get('choices', [])
                answer = json_object.get('answerKey', None)
                formatted_choices = ", ".join([f"'{choice['label']}': '{choice['text']}'" for choice in choices]) if choices else None
                question = f"{stem} Choose the correct answer from the following options: {formatted_choices}" if stem and formatted_choices else None

            elif 'Knowledge' in file_name:
                stem = json_object.get('stem', None)
                choices = json_object.get('choices', {})
                answer = json_object.get('answer', None)
                formatted_choices = ", ".join([f"'{label}': '{text}'" for label, text in choices.items()]) if choices else None
                question = f"{stem} Choose the correct answer from the following options: {formatted_choices}" if stem and formatted_choices else None

            # Ask ChatGPT the question and get the answer
            if question:
                gpt_answer = ask_chatgpt(question)
                translated_question = translate_to_korean(question)
                translated_gpt_answer = ask_chatgpt(translated_question)
                translated_back_gpt_answer = translate_to_english(translated_gpt_answer)

                # Generate embeddings for the original and translated back answers
                embedding1 = embedding_model.encode(gpt_answer).reshape(1, -1)
                embedding2 = embedding_model.encode(translated_back_gpt_answer).reshape(1, -1)

                # Compute cosine similarity
                cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]

            # Append data to lists
            questions.append(question)
            gpt_answers.append(gpt_answer)
            translated_questions.append(translated_question)
            gpt_answers_translated.append(translated_gpt_answer)
            translated_back_to_english.append(translated_back_gpt_answer)
            cosine_similarities.append(cosine_sim)
            choices_list.append(choices if choices else None)
            answers.append(answer)

    # Create DataFrame
    df = pd.DataFrame({
        'Question': questions,
        'Translated Question (Ko)': translated_questions,
        'GPT-3.5 Answer to Translated Question (Ko)': gpt_answers_translated,
        'GPT-3.5 Answer': gpt_answers,
        'Translated back GPT-3.5 Answer (En)': translated_back_to_english,
        'Answer Key' : answers,
        'Cosine Similarity' : cosine_similarities,
        'Choices' : choices_list,
    })

    # Save DataFrame
    file_name_without_extension, _ = os.path.splitext(file_name)
    df.to_csv(f'processed_{file_name_without_extension}.csv', index=False)


def main():
    # Load pre-trained sentenceBERT model
    ## https://www.sbert.net/
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

    # fetch the JSON files    
    ## https://github.com/Senyu-Li/LLM-Multilingual-Types
    file_name = 'Math_test.json'
    # file_name = 'Logic_test.json'
    # file_name = 'Knowledge_test.json'
    process_json_file(file_name, embedding_model)

if __name__ == "__main__":
    main()    