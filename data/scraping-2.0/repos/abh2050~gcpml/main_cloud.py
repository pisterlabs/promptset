import functions_framework
import openai
import json
from google.cloud import storage
from key import api_key

# Set your OpenAI API key
openai.api_key = api_key

# Initialize the Google Cloud Storage client
storage_client = storage.Client()

# Define examples for the model to understand the classification context
classification_examples = [
    {"label": "Spam", "text": "Buy now! Special offer!"},
    {"label": "Not Spam", "text": "This is a relevant discussion about the topic."}
]

# Function to classify a tweet reply
def classify_reply(reply_text):
    prompt = "Classify the following tweet reply as 'Spam' or 'Not Spam':\n\n"
    for example in classification_examples:
        prompt += f"Reply: \"{example['text']}\"\nClassification: {example['label']}\n"
    prompt += f"Reply: \"{reply_text}\"\nClassification:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=40
    )

    return response.choices[0].text.strip()

# Set your OpenAI API keyå
openai.api_key = api_key

# Initialize the Google Cloud Storage client
storage_client = storage.Client()

# Define examples for the model to understand the classification context
classification_examples = [
    {"label": "Spam", "text": "Buy now! Special offer!"},
    {"label": "Not Spam", "text": "This is a relevant discussion about the topic."}
]

# Function to classify a tweet reply
def classify_reply(reply_text):
    prompt = "Classify the following tweet reply as 'Spam' or 'Not Spam':\n\n"
    for example in classification_examples:
        prompt += f"Reply: \"{example['text']}\"\nClassification: {example['label']}\n"
    prompt += f"Reply: \"{reply_text}\"\nClassification:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=40
    )

    return response.choices[0].text.strip()

# HTTP Cloud Function for classifying tweet replies
@functions_framework.http
def classify_reply_http(request):
    # Read tweets from 'inputtweet' bucket
    input_bucket = storage_client.bucket('inputtweet')
    input_blobs = input_bucket.list_blobs()

    for blob in input_blobs:
        try:
            # Read the file content as text
            tweets_data = blob.download_as_text()

            # Parse the input JSON data
            input_data = [json.loads(line) for line in tweets_data.split('\n') if line.strip()]

            # Initialize the list to store output JSON objects
            output_data = []

            for tweet_json in input_data:
                try:
                    reply = tweet_json['reply']

                    # Classify the reply as Spam or Not Spam
                    reply_classification = classify_reply(reply)

                    # Add the classification result to the input JSON object
                    tweet_json['reply_classification'] = reply_classification

                    # Append the updated JSON object to the output list
                    output_data.append(tweet_json)

                except KeyError:
                    print("Missing 'reply' field in input JSON.")
                except json.JSONDecodeError:
                    print("Invalid JSON in input data.")

            # Write the output JSON data to 'outputtweet' bucket
            output_bucket = storage_client.bucket('outputtweet')
            output_blob = output_bucket.blob(f'classified_{blob.name}')
            output_blob.upload_from_string('\n'.join(json.dumps(obj) for obj in output_data))

        except Exception as e:
            print(f"Error processing file {blob.name}: {str(e)}")

    return 'Tweet reply classification completed.', 200
