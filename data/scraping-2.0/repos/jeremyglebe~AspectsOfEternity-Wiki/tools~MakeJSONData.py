import sys
import openai
from secrets import SECRET_API_KEY
openai.api_key = SECRET_API_KEY

# Get a filename from the command line arguments
filename = sys.argv[1]

# Read the file's contents into a string
with open(filename, "r", encoding='utf-8') as f:
    text = f.read()

# Ask GPT to format the text from the file into a JSON object or array
gpt_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": """You are a data formatter which takes plain text data and formats it into a JSON object or array.
You will be sent data by the user and reply only with the formatted JSON object or array.
(No additional text should be added to the response.)"""
        },
        {
            "role": "user",
            "content": text
        }
    ]
)

# Clean the response of special UTF-8 characters
cleaned = gpt_response.choices[0]["message"]["content"].replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

# Print the JSON object or array
print(cleaned)
