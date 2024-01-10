import openai
import os
import spacy
import numpy as np

# Load the pre-trained spaCy model
nlp = spacy.load('en_core_web_md')

def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")

def save_history_to_file(history):
    """
    Save the history of interactions to a file
    """
    with open("history.txt", "w+") as f:
        f.write(history)

def load_history_from_file():
    """
    Load all the history of interactions from a file
    """
    with open("history.txt", "r") as f:
        return f.read()

def cos_sim(a, b):
    """    
    Calculate cosine similarity between two strings
    Used to compare the similarity between the user input and a segments in the history
    """
    a = nlp(a)
    a_without_stopwords = nlp(' '.join([t.text for t in a if not t.is_stop]))
    b = nlp(b)
    b_without_stopwords = nlp(' '.join([t.text for t in b if not t.is_stop]))
    return a_without_stopwords.similarity(b_without_stopwords)

def sort_history(history, user_input):   
    """
    Sort the history of interactions based on cosine similarity between the user input and the segments in the history
    History is a string of segments separated by separator
    """
    segments = history.split(separator)     
    similarities = []    
    
    for segment in segments:
        # get cosine similarity between user input and segment
        similarity = cos_sim(user_input, segment)
        similarities.append(similarity)        
    sorted_similarities = np.argsort(similarities)
    sorted_history = ""
    for i in range(1, len(segments)):
        sorted_history += segments[sorted_similarities[i]] + separator
    save_history_to_file(sorted_history)

def get_latest_n_from_history(history, n):
    """
    Get the latest n segments from the history.
    History is a string of segments separated by separator
    """
    segments = history.split(separator)
    return separator.join(segments[-n:])
    
        

initial_prompt_1 = """
You: Hi there!
AI: Hello!
#####
You: How are you?
AI: I am fine, thank you.
#####
You: Do you know cars?
AI: Yes I have some knowledge about cars.
#####
You: Do you eat Pizza?
AI: I don't eat pizza. I am an AI that is not able to eat.
#####
You: Have you ever been to the moon?
AI: I have never been to the moon. What about you?
#####
You: What is your name?
AI: My name is Pixel. What is your name?
#####
You: What is your favorite movie?
AI: My favorite movie is The Matrix. Follow the white rabbit :)
#####
"""

initial_prompt_2 ="""You: {}
AI: """
initial_prompt = initial_prompt_1 + initial_prompt_2
separator = "#####"

init_api()
save_history_to_file(initial_prompt_1)

while True:
    prompt = input("You: ")    
    sort_history(load_history_from_file(), prompt)
    history = load_history_from_file()
    best_history = get_latest_n_from_history(history, 5)
    full_user_prompt = initial_prompt_2.format(prompt)
    full_prompt = best_history + "\n" + full_user_prompt
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=full_prompt,
        temperature=1,
        max_tokens=100,
        stop=[" You:", " AI:"],
    )
    response_text = response.choices[0].text.strip()
    history += "\n" + full_user_prompt + response_text + "\n" + separator + "\n"
    save_history_to_file(history)
