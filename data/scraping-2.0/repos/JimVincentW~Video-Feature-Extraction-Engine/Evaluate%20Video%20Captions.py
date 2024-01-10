import openai
import json

# Load the OpenAI API key
openai.api_key = "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"

# Load the captions from the JSON file
with open("output.json", "r") as f:
    captions = json.load(f)["captions"]

# Combine the captions into a single string
prompt_lines = "\n".join(captions)

MODEL = "gpt-3.5-turbo-0301"

response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are aiding in categorizing Videos." },
        {"role": "user", "content": "The following Lines are the captions that Computer Vision Information Retrieval model outputs. The Intention is to categorize and label the video. Provide 5 hashtags for it. Just the regular ones of a social media app. Also Place it into one of the categories: 1. Sports , 2. User-generated content, Private Event, 5. Outside with people, 6. inside of the appartment . Explain."},
        {"role": "assistant", "content": "Okay, so what are the captions?"},
        {"role": "user", "content": "Captions:\n\n" + prompt_lines},
        {"role": "system", "content": "Now filter everything tha is not a hashtag and more likely just because the vision model just randomly picked up on it."},
    ],
    temperature=0,
)

print(response.choices[0]["message"]["content"])