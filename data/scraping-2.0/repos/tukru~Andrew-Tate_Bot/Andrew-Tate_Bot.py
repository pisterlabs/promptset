import openai

# Set up OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

# Read the training data from tate-data.json
with open('tate-data.json', 'r') as file:
    training_data = file.read()

# Fine-tune the model
response = openai.FineTunes.create(
    model="text-davinci-003",  # Use GPT-3.5 Turbo
    training_data=training_data,
    prompt_language="en",
    n=1,  # Number of checkpoints to create
    display_name="Tate-Bot Fine-Tuned Model"
)

# Access the fine-tuned model ID
model_id = response.checkpoint_id

# Print the fine-tuned model ID
print("Fine-tuned model ID:", model_id)
