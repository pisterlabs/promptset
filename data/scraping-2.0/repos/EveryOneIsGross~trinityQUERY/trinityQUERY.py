import openai
from openai import ChatCompletion
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import json
import dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

dotenv.load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

data = []

# Load existing data from JSON file
if os.path.exists("chatbot_data.json"):
    with open("chatbot_data.json", "r") as infile:
        data = json.load(infile)
else:
    data = []

def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked_keywords = r.get_ranked_phrases()
    keywords = [word for word in word_tokenize(ranked_keywords[0])] if len(ranked_keywords) > 0 else []
    print(f"Extracted keywords: {keywords}")

    # Filter keywords to keep only relevant words
    relevant_keywords = filter_relevant_keywords(keywords)

    return relevant_keywords[0] if len(relevant_keywords) > 0 else ""

def filter_relevant_keywords(keywords):
    relevant_keywords = []
    for keyword in keywords:
        if wordnet.synsets(keyword):
            relevant_keywords.append(keyword)
    return relevant_keywords

def calculate_sentiment_score(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    return sentiment_scores

def chatbot(query, bot_name, role, parent_response=None, temperature=0.5, max_tokens=200, top_p=0.2, frequency_penalty=0.2, presence_penalty=0.0): 
    keywords = extract_keywords(query)
    model = "gpt-3.5-turbo-16k"
    messages = [{"role": "system", "content": role},
                  {"role": "user", "content": query}]
    if parent_response:
        messages.append({"role": "user", "content": parent_response})
    response = ChatCompletion.create(
        model=model, 
        messages=messages,
        temperature=temperature,  # Controls randomness. Higher values (closer to 1) make output more random.
        max_tokens=max_tokens + 200,  # Maximum length of the output in tokens (words).
        top_p=top_p,  # Nucleus sampling: smaller values make the output more focused.
        frequency_penalty=frequency_penalty,  # Penalizes new tokens based on their frequency.
        presence_penalty=presence_penalty,  # Penalizes new tokens based on their presence in the input.
    )
    response_text = response['choices'][0]['message']['content']
    response_keywords = extract_keywords(response_text)
    
    # Calculate sentiment score
    sentiment_score = calculate_sentiment_score(response_text)
    
    # Add the information to the data dictionary
    data.append({
        "bot": bot_name,
        "keywords": response_keywords,
        "input": query,
        "response": response_text,
        "role": role,
        "sentiment_score": sentiment_score
    })
    
    return response_text


def chatbot_system(query, structure='v', role_a="You are bot a.", role_b="You are bot b.", role_c="You are bot c."):
    print(f"Structure chosen: {structure}")

    if structure == 'v':
        print("Running 'v' structure...")
        response_a = chatbot(query, "Chatbot A", role_a)
        response_b = chatbot(query, "Chatbot B", role_b)
        response_c = chatbot(query, "Chatbot C", role_c, response_a + " " + response_b)
        return response_c
    
    elif structure == 'confounder':
        print("Running 'confounder' structure...")
        response_c = chatbot(query, "Chatbot C", role_c)
        response_a = chatbot(query, "Chatbot A", role_a, response_c)
        response_b = chatbot(query, "Chatbot B", role_b, response_c)
        return response_a, response_b
    
    elif structure == 'chain':
        print("Running 'chain' structure...")
        response_a = chatbot(query, "Chatbot A", role_a)
        response_b = chatbot(query, "Chatbot B", role_b, response_a)
        response_c = chatbot(query, "Chatbot C", role_c, response_b)
        return response_c
    
    else:
        raise ValueError("Invalid structure: must be 'v', 'confounder', or 'chain'")

# Get user input
query = input("Enter your question: ")
role_a = input("Enter the role for Chatbot A: ")
role_b = input("Enter the role for Chatbot B: ")
role_c = input("Enter the role for Chatbot C: ")
structure = input("Choose the structure ('v', 'confounder', or 'chain'): ")

# Run the chatbot system
response = chatbot_system(query, structure, role_a, role_b, role_c)

# Create a new data entry for the current execution
execution_data = {
    "structure": structure,
    "user_query": query,
    "final_response": response,
    "bot_roles": {
        "Chatbot A": role_a,
        "Chatbot B": role_b,
        "Chatbot C": role_c
    }
}

# Append the new data to the existing data
data.append(execution_data)

# Save the updated data as JSON
with open("chatbot_data.json", "w") as outfile:
    json.dump(data, outfile, indent=4)

print("Final response:", response)
