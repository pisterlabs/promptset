from openai import OpenAI
from dotenv import load_dotenv
import time, os

# Load the .env file
load_dotenv('.env')

# Get the environment variables
ASSISTANT_ID = os.getenv('ASSISTANT_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_answer_from_assistant(prompt, assistant_id):
    """
    Sends a prompt to the OpenAI assistant and returns its response.

    Args:
        prompt (str): The prompt to send to the assistant.

    Returns:
        str: The assistant's response.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role='user', 
        content=prompt
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

    start_t = time.time()
    while time.time() - start_t < 20:

        client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        messages = client.beta.threads.messages.list(thread_id=thread.id)

        if messages.data[0].role == "assistant" and len(messages.data[0].content[0].text.value) > 0:

            break
    
    return messages.data[0].content[0].text.value

def identify_book_from_prompt(prompt):
    """
    Identifies a book based on a prompt.

    Args:
        prompt (str): The prompt to identify the book from.

    Returns:
        dict: A dictionary containing the book's title, author, summary, and alternative title.
    """
    answer = get_answer_from_assistant(prompt, ASSISTANT_ID)

    return get_book_info_from_answer(answer)

def get_book_info_from_answer(answer):
    """
    Extracts book information from an assistant's answer.

    Args:
        answer (str): The assistant's answer.

    Returns:
        dict: A dictionary containing the book's title, author, summary, and alternative title.
    """
    book_info = {
        'title': '',
        'author': '',
        'summary': '',
        'title_alternative': ''
    }

    parts = answer.split('#')

    for part in parts:
        if part.startswith('Title:'):
            book_info['title'] = part.replace('Title: ', '').strip()
        elif part.startswith('Title-'):
            book_info['title_alternative'] = part.split(':')[-1][1:]
        elif part.startswith('Author:'):
            book_info['author'] = part.replace('Author: ', '').strip()
        elif part.startswith('Summary:'):
            book_info['summary'] = part.replace('Summary: ', '').strip()

    return book_info

def get_chatgpt_recommendations(book_title, book_author, number_of_recommendations=4):
    """
    Gets book recommendations based on a given book's title and author.

    This function sends a prompt to the OpenAI assistant asking for book recommendations 
    based on the given book's title and author. The assistant's response is then parsed 
    and the recommended books' information is extracted and returned.

    Args:
        book_title (str): The title of the book.
        book_author (str): The author of the book.
        number_of_recommendations (int, optional): The number of recommendations to get. Defaults to 4.

    Returns:
        list: A list of dictionaries, where each dictionary contains the title, author, 
              summary, and alternative title of a recommended book.
    """
    prompt = f"Give me {number_of_recommendations} recommendations if I read {book_title} by {book_author}"

    answer = get_answer_from_assistant(prompt)

    rec_answer = answer.split('*')

    recommendations = []

    for rec in rec_answer[1:]:
        
        book_info = {
            'title': '',
            'author': '',
            'summary': '',
            'title_alternative': ''
        }
        
        parts = rec.split('#')

        for part in parts:
            if part.startswith('Title:'):
                book_info['title'] = part.replace('Title: ', '').strip()
            elif part.startswith('Title-'):
                book_info['title_alternative'] = part.split(':')[-1][1:]
            elif part.startswith('Author:'):
                book_info['author'] = part.replace('Author: ', '').strip()
            elif part.startswith('Summary:'):
                book_info['summary'] = part.replace('Summary: ', '').strip()

        recommendations.append(book_info)

    return recommendations

