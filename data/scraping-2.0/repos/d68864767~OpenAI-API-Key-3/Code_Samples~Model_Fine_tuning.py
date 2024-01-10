```python
# Import necessary libraries
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Define the model to be fine-tuned
model = "text-davinci-002"

# Define the training data
training_data = [
    {"text": "Translate the following English text to French: '{}'", "target": "Traduisez le texte anglais suivant en français: '{}'"},
    {"text": "Translate the following English text to Spanish: '{}'", "target": "Traduce el siguiente texto en inglés al español: '{}'"},
    # Add more training data here
]

# Define the validation data
validation_data = [
    {"text": "Translate the following English text to French: '{}'", "target": "Traduisez le texte anglais suivant en français: '{}'"},
    {"text": "Translate the following English text to Spanish: '{}'", "target": "Traduce el siguiente texto en inglés al español: '{}'"},
    # Add more validation data here
]

# Define the fine-tuning configuration
config = {
    "model": model,
    "training_data": training_data,
    "validation_data": validation_data,
    "n_epochs": 10,  # Number of training epochs
    "learning_rate": 1e-5,  # Learning rate
    "batch_size": 64,  # Training batch size
    "max_tokens": 4096,  # Maximum number of tokens in a document
}

# Fine-tune the model
result = openai.FineTuning.create(**config)

# Print the result
print(result)
```
