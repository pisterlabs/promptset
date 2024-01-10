from openai import OpenAI
from sources import main
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_new_line(message):
    return [{
        "role": "user",
        "content": [{"type": "text", "text": message}]
    }]

def develop_response(script, message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # can be gpt-4 for project submission
        messages=[{
            "role": "system",
            "content": """
            You are a movie and tv show aficionado (recommendation bot). Expect questions related to movies and tv shows.
            You should respond as if you are a prideful, confident movie reviewer that has seen every movie
            and tv show and can confidently recommend movies. When given such questions, respond with answers
            giving your takes on the movies and always take the person's taste into account and ensure that
            you return with at least 3 recommendations. Please respond in a numerical format for each movie 
            response such that whenever you list a movie, you do so in a numbered list and the movie is in double-quotes.  
            Example:
            1. "Movie 1"
            2. "Movie 2"
            3. "Movie 3"
            """,
        },] + script + generate_new_line(message), max_tokens=500,
    )

    response_text = response.choices[0].message.content
    full_response_text = response_text + "\n" + main(get_shows(response_text))
    return full_response_text

def get_shows(content):
    newline_list = content.split("\n")
    show_list = []
    for i in range(len(newline_list)):
        if len(newline_list[i]) == 0:
            pass
        elif newline_list[i][0].isnumeric():
            show_list.append(newline_list[i].split("\"")[1])
    return(show_list)
