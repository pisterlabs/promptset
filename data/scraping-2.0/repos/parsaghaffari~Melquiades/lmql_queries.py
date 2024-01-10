import os
import lmql

from config import OPENAI_API_KEY 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

@lmql.query(model="openai/gpt-3.5-turbo-instruct")
def get_characters(book, author="", num_chars=5):
    '''lmql
    """Answering the following questions about the book {book} by {author}:

    Here's a list of major characters from the book: \n"""
    chars=[]
    for i in range(num_chars):
        "-[CHARACTER]" where STOPS_AT(CHARACTER, "\n")
        chars.append(CHARACTER.strip())
    return chars
    '''

@lmql.query(model="gpt-4")
def get_character_description(character, book, author):
    '''lmql
    """Here's an accurate and concise visual description of {character} from {book} by {author} which can be used to paint their portrait, broken down into face, hair, expression, attire, accessories, and background (don't use the words 'thick' or 'tied up' or 'bare' or 'bathing'): [DESCRIPTION]"""
    '''