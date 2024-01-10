import subprocess
import openai
import os

# Set the path to your API key file
api_key_file = "api_key.txt"

# Read the API key from the file
with open(api_key_file, "r") as f:
    api_key = f.read().strip()

# Set the environment variable for the API key
env = dict(os.environ)
env["OPENAI_API_KEY"] = api_key

# Define the path to your training data file
train_file_path = "training_data/training_data.jsonl"

# Define the base model you want to start from
base_model = "curie"

# Define the suffix for your fine-tuned model (optional)
suffix = "my_finetuned_model"

# Create the fine-tuning job
create_command = ["openai", "api", "fine_tunes.create", "-t", train_file_path, "-m", base_model, "--suffix", suffix]

# Run the create fine-tunes command
result = subprocess.check_output(create_command, env=env, text=True)
print(result)
