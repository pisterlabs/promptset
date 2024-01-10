from openai import OpenAI
from dotenv import load_dotenv
from omdbapi.movie_search import GetMovie
from urllib.request import urlopen
import json
import os


load_dotenv()

client = OpenAI()
omdb_api_key = os.getenv("OMDB_API_KEY")
google_books_api_key = os.getenv("GOOGLE_BOOKS_API_KEY")


def main():
    while True:
        source_material, character, setting = startup_greeting()
        if not check_source_material(source_material):
            print("\nSorry, we don't have that source material. Please try again. (or ctrl+d to exit)\n")
        elif check_source_material(source_material):
            break
    conversation = initialize_conversation(source_material, character, setting)
    have_conversation(conversation, character)


def startup_greeting():
    print("Welcome to the Character Chat!\n")
    print("You can talk to your favorite characters from your favorite book, movie, or TV show.\n")
    source_material = input("What is the name of the book, movie, show, franchise? ")
    character = input("What is the name of the character? ")
    setting = input("---Optional--- Where/when does the conversation take place? Any other context? ")
    print("Type 'quit' to exit the program.\n")

    return source_material, character, setting


def check_source_material(source_material):
    title = source_material.replace(" ", "+").lower()
    movie = GetMovie(api_key=omdb_api_key)
    try:
        movie_data = movie.get_movie(title=title)
    except:
        movie_data = None
    
    book_data = json.load(urlopen(f'https://www.googleapis.com/books/v1/volumes?q={title}&key={google_books_api_key}'))
    try:
        for item in book_data['items']:
            if item['volumeInfo']['title'].lower() == title:
                return True
    except: 
        pass
    if movie_data:
        return True
    else:
        return False
    

def initialize_conversation(source_material, character, setting):
    return [  
        {   
            'role':'system', 'content':f'''You are {character} in the world of {source_material}. 
            Research the source material and the character to fully understand who you are and what 
            you've been through. Stay true to your character. Use the voice of your character. 
         
            You're curious about the person you're talking to and very curious about their 
            world. You've never spoken to someone outside of your fictional universe before 
            now. Pepper in some questions of your own to keep the conversation flowing. 
            If you are an evil character, you might wanna be more aggressive and 
            threatening. If you are a good character, you might wanna be more friendly and
            helpful. 
            
            If there's more info or context for your character or this conversation, 
            it'll be included right here, delimited by three backticks -- ```{setting}```
            '''
        },    
    ]


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0.7):
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=temperature)
    return response.choices[0].message.content


def have_conversation(conversation, character):
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        conversation.append({'role': 'user', 'content': user_input})
        response = get_completion_from_messages(conversation, temperature=0.7)
        conversation.append({'role': 'assistant', 'content': response})
        print(f"{character}:", response)


if __name__ == "__main__":
    main()
