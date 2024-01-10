import openai
from transformers import pipeline
from transformers import AutoTokenizer
from var_dump import var_dump
import tiktoken
import mneumosyne

def import_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except IOError:
        print(f"Error reading file: {file_path}")
        return None


def indexTokenizer(inputText):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(inputText)
    return tokens


memory = mneumosyne.LongTermMemory()



openAPIKey = import_text_file("../openapikey.txt")
# Replace 'YOUR_OPENAI_API_KEY' with your actual API key from OpenAI
openai.api_key = openAPIKey


def get_sentiment_score(text):
    # Initialize the sentiment analyzer
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    # Get the sentiment score of the input text
    sentiment_score = sentiment_analyzer(text)[0]
    return sentiment_score


def generate_gpt3_response(prompt, sentiment_score):
    # Define the GPT-3 API parameters based on the sentiment score
    if sentiment_score['label'] == 'POSITIVE':
        # If the sentiment score is positive, use a setting for a positive response
        gpt3_parameters = {
            "temperature": 0.7,
            "max_tokens": 150
        }
    else:
        # If the sentiment score is negative or neutral, use a setting for a more general response
        gpt3_parameters = {
            "temperature": 0.8,
            "max_tokens": 200
        }

    enc = tiktoken.encoding_for_model("text-davinci-002")
    print("tokens: ")
    print(enc.encode(prompt))

    # Generate the GPT-3 response using the given parameters
    response = openai.Completion.create(
        engine="text-davinci-002",  # Replace with the engine you want to use
        prompt=prompt,
        **gpt3_parameters
    )

    return response['choices'][0]['text']


# Example usage:
user_input = "Can you explain the movie 'The Last Unicorn' to me?"
sentiment_score = get_sentiment_score(user_input)
print("Sentiment Score:", sentiment_score)

gpt3_response = generate_gpt3_response(user_input, sentiment_score)
# var_dump(gpt3_response)
print("GPT-3 Response:")
print(gpt3_response)
