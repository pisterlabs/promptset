import openai
from openai import OpenAI

# Function to open a file and return its contents as a string
def open_file(filepath):
    """Read the contents of a file and return it as a string."""
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to save content to a file
def save_file(filepath, content):
    """Save the given content to a file, appending to any existing content."""
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

# Read the OpenAI API key from a file and set it for the client
api_key = open_file('openaiapikey2.txt')
client = OpenAI(api_key=api_key)

# Define the file ID for the training data and the model name
training_file_id = "file-7pnaypfeo8LHoO6xbHccXNTz"
model_name = "gpt-3.5-turbo-1106"  # Or another base model if you prefer

# Create the fine-tuning job
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    model=model_name
)

# Access the job ID using dot notation and print it
job_id = response.id
print(f"Fine-tuning job created successfully with ID: {job_id}")
