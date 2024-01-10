import openai

# Define a function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Define a function to save content to a file
def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)

# Set the OpenAI API keys by reading them from files
api_key = open_file('openaiapikey2.txt')

openai.api_key = api_key

# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve("YOUR FT JOB ID")
status = response['status']
print(f"Fine-tuning job status: {status}")