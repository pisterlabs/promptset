# Combining the logic from both scripts

# Import necessary libraries
import json
import openai
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import dotenv
import os

# Load the .env file
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class State:
    def __init__(self, thought, feeling, behavior, memory):
        self.thought = thought
        self.feeling = feeling
        self.behavior = behavior
        self.memory = memory

    def save(self):
        with open('state.json', 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls):
        with open('state.json', 'r') as f:
            data = json.load(f)
            return cls(**data)

# Sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Function to compare responses
def compare_responses(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]


def chatbot(message, state):
    # Thought generation
    # Here, we are assuming that the "message" parameter is the user query.
    prompt = f"User query: {message}\\n\\nSome thoughts on this query are:\\n"
    completion = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
        stop="\\n\\n"
    )
    thought = completion["choices"][0]["text"]
    state.thought = thought

    # Sentiment analysis for feeling determination
    state.feeling = analyze_sentiment(thought)

    # Behavior adjustment
    thought_sentiment = analyze_sentiment(state.thought)
    feeling_sentiment = analyze_sentiment(state.feeling)

    if thought_sentiment == feeling_sentiment:
        state.behavior += 0.1
    else:
        state.behavior -= 0.1

    # Message generation
    state_representation = f"Thought: {state.thought}, Feeling: {state.feeling}, Behavior: {state.behavior}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": state_representation},
            {"role": "user", "content": message}
        ],
        temperature=state.behavior
    )

    response_content = response.choices[0].message.content
    match_score = compare_responses(message, response_content)
    state.memory.append({"message": message, "response": response_content, "match_score": match_score})
    state.save()

    return response_content



def main():
    # Initialize a default state for the agent
    state = State(thought="", feeling="", behavior=0.5, memory=[])

    print("Welcome to the CBT Chatbot! Type 'exit' or 'quit' to end the conversation.")
    while True:
        # Get user input
        message = input("You: ")
        
        # Check if user wants to exit
        if message.lower() in ["exit", "quit"]:
            print("Exiting the chat. Have a great day!")
            break
        
        # Get response from chatbot and display it
        response = chatbot(message, state)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()

