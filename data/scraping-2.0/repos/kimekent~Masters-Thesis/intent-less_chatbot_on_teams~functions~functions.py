"""
This file contains the functions that are used to run the intent-less chatbot with Microsoft Teams front end.
(located at '\intent-less_chatbot_on_Teams\bots\echo-bot.py')
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


import smtplib
import ssl
from email.message import EmailMessage

def send_email(query):
    """
    Send an email with the provided query as the body.
    This function is used to send user query as ticket when OpenAI Websupport-Bot can't answer question.

    :param query: The content to be sent in the email body.
    """
    email_body = query
    email_sender = "kimberly.kent.twitter@gmail.com"
    email_password = "mihtepktfliycluz"
    email_receiver = "mefabe7562@marksia.com"
    subject = "Question from Websupportbot"

    em = EmailMessage()
    em["From"] = email_sender
    em["To"] = email_receiver
    em["Subject"] = subject
    em.set_content(email_body)

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())


def adjust_similarity_scores(results, question, word_intent_dict, multiplier):
    """
    Adjust the similarity scores of documents based on the presence of specific words.

    :param results: A list of tuples containing (document, similarity_score)
    :param words_to_check: A list of words to search for in the question
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