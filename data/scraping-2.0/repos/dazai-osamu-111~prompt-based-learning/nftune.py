import openai
from openai import OpenAI

# Define a function to open a file and return its contents as a string
def open_file(filepath):
    """Open a file and return its contents as a string."""
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Define a function to save content to a file
def save_file(filepath, content):
    """Save content to a file."""
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

# Set the OpenAI API keys by reading them from files
api_key = open_file('openaiapikey2.txt')

# Initialize OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

# Path to the training data
training_data_path = "PATH/schema.jsonl"

# Upload the training file using the Files API
with open(training_data_path, "rb") as file:
    response = client.files.create(
        file=file,
        purpose="fine-tune"
    )

# Instead of using response['id'], use response.id to access the file ID
file_id = response.id
print(f"File uploaded successfully with ID: {file_id}")
