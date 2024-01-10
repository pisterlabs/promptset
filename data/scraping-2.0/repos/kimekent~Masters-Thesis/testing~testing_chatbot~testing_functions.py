"""
This file contains the functions used to run the testing OpenAI chatbots located at '\testing\testing_chatbot'.
"""

from time import time, sleep
import openai

def gpt3_1106_completion(prompt, model='gpt-3.5-turbo-1106', temperature=0.7, max_tokens=1000, log_directory=None):
    """
    Generate text using OpenAI's gpt-3.5-turbo-1106 model and log the response.
    This generation function is used to generate answers to users websupport quesitons.

    :param prompt: The input text to prompt the model.
    :param model: The GPT-3.5 model used for generating text (default 'gpt-3.5-turbo-1106').
    :param temperature: The temperature setting for response generation (default 0.7).
    :param max_tokens: The maximum number of tokens to generate (default 1000).
    :param log_directory: Directory to save the generated responses.
    :return: The generated completion text.
    """
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            try:
                # Extract and save the response
                text = response.choices[0].message.content.strip()
                filename = f'{time()}_gpt3.txt'
                with open(f'{log_directory}/{filename}', 'w') as outfile:
                    outfile.write('PROMPT:\n\n' + prompt + '\n\n====== \n\nRESPONSE:\n\n' + text)
                return text
            except:
                print("error saving to log")

        except Exception as e:
            # Handle errors and retry
            retry += 1
            if retry > max_retry:
                return f"GPT3 error: {e}"
            print('Error communicating with OpenAI:', e)
            sleep(1)


def open_file(filepath):
    """
    Open and read the content of a file.

    :param filepath: The path of the file to be read.
    :return: The content of the file as a string.
    """
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(file_content, destination_file):
    """
    Save the given content into a file.

    :param file_content: The content to be written to the file.
    :param destination_file: The path of the file where the content will be saved.
    """
    with open(destination_file, 'w', encoding='utf-8') as outfile:
        outfile.write(file_content)


import re

def remove_history(text, pattern_to_replace, pattern_to_add):
    """
    Remove a specific pattern from the text and replace it with another pattern.
    This function is used to remove the history from the retriever and answer prompt when new chatbot
    session start.

    :param text: The original text with the pattern to be replaced.
    :param pattern_to_replace: The regex pattern to find and remove in the text.
    :param pattern_to_add: The pattern to replace the removed text with.
    :return: The modified text with the pattern replaced.
    """
    pattern = r"CHAT-HISTORY:(.*?)<<history>>"
    return re.sub(pattern_to_replace, pattern_to_add, text, flags=re.DOTALL).strip()


import tiktoken

def num_tokens_from_string(text, encoding):
    """
    Calculate the number of tokens in a text string based on the specified encoding.

    :param text: The text string to be tokenized.
    :param encoding: The encoding to be used for tokenization.
    :return: The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding(encoding)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def replace_links_with_placeholder(text):
    """
    Replace URLs in a given text with a placeholder.
    This function is used to remove urls from the websupport questions, since they hold no value.

    :param text: The text containing URLs to be replaced.
    :return: The text with URLs replaced by a placeholder '<<link>>'.
    """
    # Define a regular expression pattern to match URLs
    url_pattern = r'https://\S+'
    modified_link = re.sub(url_pattern, '<<link>>', str(text))
    return modified_link


import numpy as np

def similarity(v1, v2):
    """
    Calculate the similarity score between two vectors as their dot product.

    :param v1: The first vector (numpy array or a list).
    :param v2: The second vector (numpy array or a list).
    :return: The dot product of the two vectors, representing their similarity.
    """
    return np.dot(v1, v2)


def gpt3_embedding(content, engine="text-embedding-ada-002"):
    """
    Generate an embedding for the given content using OpenAI's embedding model.

    :param content: The text content to generate an embedding for.
    :param engine: The OpenAI embedding model to use. Default is 'text-embedding-ada-002'.
    :return: A vector representing the embedding of the input content.
    """
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


import json

def OpenAI_retriever(query, count=20, index_path=None,
                     word_intent_dict={"word":("intent",)}, multiplier=1.2):
    """
    Retrieve and rank documents based on their similarity to a query using OpenAI embeddings.
    The higher the score, the similar the query and document. Score ranges from 0 - 1.

    :param query: The query text to compare against documents.
    :param count: The number of top matching documents to return. Default is 20.
    :param index_path: The path to the JSON file containing pre-computed embeddings and associated metadata.
    :return: A list of the top 'count' documents sorted by similarity to the query, including context, score, and metadata.
    """
    vector = gpt3_embedding(query)
    with open(index_path, 'r') as infile:
        data = json.load(infile)
    scores = []
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'context': i['context'], 'score': score, "metadata": i["metadata"]})
    # Comment for unadjusted scores: If certain words or word combinations in the question indicate a specific intent,
    # the scores of documents associated with that intent are multiplied. This increases
    # the likelihood of selecting documents relevant to the question.
    scores = adjust_similarity_scores(results=scores, question=query, word_intent_dict=word_intent_dict, multiplier=multiplier)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]


def adjust_similarity_scores(results, question, word_intent_dict, multiplier):
    """
    Adjust the similarity scores of documents based on the presence of specific words in a question,
    and their corresponding intents.

    :param results: A list of dictionaries, each containing 'context', 'score', and 'metadata'.
    :param question: The question text to search for specific words.
    :param word_intent_dict: A dictionary mapping words to a tuple of corresponding intents.
    :param multiplier: The multiplier to apply to the score if the document's intent matches.
    :return: Adjusted list of documents with all metadata and adjusted_similarity_score.
    """
    adjusted_results = []

    # Extract words from the question
    question_lower = question.lower()

    # Determine relevant intents based on single words and combinations
    relevant_intents = set()
    for key, intents in word_intent_dict.items():
        if isinstance(key, tuple):  # Check for word pairs or sequences
            if all(word.lower() in question_lower for word in key):
                relevant_intents.update(intents)
        elif isinstance(key, str):  # Check for single words
            if key.lower() in question_lower:
                relevant_intents.update(intents)

    for document in results:
        try:
            doc_intent = document["metadata"]["intent"]
            if doc_intent in relevant_intents:
                document['score'] *= multiplier
                document['metadata']['adjusted_similarity_score'] = document['score']
            adjusted_results.append(document)

        except KeyError:
            print(e)
            continue

    adjusted_results.sort(key=lambda x: x['score'])
    return adjusted_results

def adjust_similarity_scores_final_model_test(results, question, word_intent_dict, multiplier):
    """
    Adjust the similarity scores of documents based on the presence of specific words.

    :param results: A list of tuples containing (document, similarity_score)
    :param words_to_check: A list of words to search for in the documents
    :param multiplier: The multiplier to apply to the score for each word found
    :return: Adjusted list of documents with all metadata and adjusted_similarity_score
    """
    adjusted_results = []

    # Extract words from the question
    question_lower = question.lower()

    # Determine relevant intents based on single words and combinations
    relevant_intents = set()
    for key, intents in word_intent_dict.items():
        if isinstance(key, tuple):  # Check for word pairs or sequences
            if all(word.lower() in question_lower for word in key):
                relevant_intents.update(intents)
        elif isinstance(key, str):  # Check for single words
            if key.lower() in question_lower:
                relevant_intents.update(intents)
    for document, score in results:
        try:
            doc_intent = document.metadata.get("intent")
            if doc_intent in relevant_intents:
                adjusted_score = score * multiplier
                document.metadata['adjusted_similarity_score'] = adjusted_score
            else:
                document.metadata['adjusted_similarity_score'] = score
            adjusted_results.append((document, score))

        except KeyError as e:
            print(e)
            continue

    adjusted_results.sort(key=lambda x: x[1])

    return adjusted_results