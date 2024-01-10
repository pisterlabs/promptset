import openai
import os

# Set up OpenAI API credentials
openai.api_key = "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"

# Define the path to your training data file
training_data_path = "C:\Davidlocal\PMD_AI_TRANNING2\exampleconversation2.txt"

# Load the training data from the file
with open(training_data_path, "r", encoding="utf-8") as f:
    training_data = f.read().splitlines()

# Define the name and other parameters for your new model
model_name = "PMDtraining1"
language = "en"
completions_per_second = 10
max_tokens = 1000
ngram_size = 3
batch_size = 1

# Create the new model with the training data
model = openai.Completion.create(
    display_name=model_name,
    language=language,
    completions_per_second=completions_per_second,
    max_tokens=max_tokens,
    ngram_size=ngram_size,
    batch_size=batch_size,
    training_data=[{"text": text} for text in training_data]
)

# Print the ID of the new model
print(f"Created new model with ID {model.id}")
