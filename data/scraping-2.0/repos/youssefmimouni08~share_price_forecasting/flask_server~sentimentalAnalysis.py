import openai
import json


# Open the JSON file
with open('../config/default.json', 'r') as file:
  # Parse the JSON data
  data = json.load(file)

# Access the value you want
value = data['openai_key']

# Set the OpenAI API key
openai.api_key = value

def classify_sentiment(sentence):
    # Use the OpenAI API to classify the sentiment of the sentence
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Is the sentiment of the following sentence positive or negative based on the topic of oil market ? {sentence}",
        max_tokens=1024
    )
    # Extract the classification from the response
    classification = response["choices"][0]["text"].strip()
    classification = classification.lower()  # Convert the string to lowercase
    if "positive" in classification:
        return "positive"
    elif "negative" in classification:
        return "negative"
    return "neutral"