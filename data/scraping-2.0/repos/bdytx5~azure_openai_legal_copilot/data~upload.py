import openai
import os

openai.api_key = "your api key" 
openai.api_base = "your endpoint URL"

openai.api_type = 'azure'
openai.api_version = '2023-09-15-preview' # Required API version for fine-tuning

# File names for training and validation datasets
training_file_name = './legalbench_subsets_train.jsonl'
validation_file_name = './legalbench_subsets_val.jsonl'

# Upload training dataset file
training_response = openai.File.create(
    file=open(training_file_name, "rb"), purpose="fine-tune", user_provided_filename="legalbench_subsets_train.jsonl"
)
training_file_id = training_response["id"]

# Upload validation dataset file
validation_response = openai.File.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune", user_provided_filename="legalbench_subsets_val.jsonl"
)
validation_file_id = validation_response["id"]

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)
