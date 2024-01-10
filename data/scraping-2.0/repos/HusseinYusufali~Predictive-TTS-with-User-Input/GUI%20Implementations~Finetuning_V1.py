import openai
import math
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

# Set up OpenAI API key

api_key = < >

openai.api_key = api_key


# Load custom dataset
with open('/Users/husseinyusufali/Desktop/PhD/Main PhD Folder/PhD - Year 2/Technical System Implementation/Predictive TransformerTTS/Data/AAC Dataset/sent_test_aac.txt', 'r') as f:
    data = f.read().splitlines()

# Define fine-tuning task
task = "predict_next_phrase"

# Set training parameters
params = {
    "model": "text-davinci-003",
    "prompt": "",
    "temperature": 0.5,
    "max_tokens": 10,
    "n": 1,
}

# Define training epochs and learning rate
epochs = 3
lr = 1e-5

progress_bar = tqdm(total=epochs, desc='Progress', unit='iteration')

for i in range(epochs):

    # Define lists to store loss and perplexity values for each epoch
    epoch_loss = []
    epoch_perplexity = []

    # Define list to store training data dictionaries
    training_data = []

    # Fine-tune the GPT-3 model on custom dataset
    for epoch in range(epochs):
        current_loss = 0
        current_perplexity = 0
        for text in data:
            params['prompt'] = task + ": " + text
            response = openai.Completion.create(
                engine=params['model'],
                prompt=params['prompt'],
                max_tokens=params['max_tokens'],
                n=params['n'],
                temperature=params['temperature'],
                stop=None,  # Disable early stopping
                logprobs=0,  # Request log probabilities for each token
            )
            completions = response.choices[0].text.strip()
            if completions != "":
                training_data.append({
                    "text": params['prompt'],
                    "completions": completions,
                    "id": str(epoch)
                })

            # Calculate loss and perplexity
            tokens = completions.split()
            logprobs = response.choices[0].logprobs.token_logprobs[1:]  # Exclude the prompt tokens
            if logprobs:
                loss = -sum(logprobs) / len(logprobs)
                perplexity = math.exp(loss)
            else:
                loss = 0
                perplexity = 0

            current_loss += loss
            current_perplexity += perplexity

        # Normalize the loss and perplexity values by dividing by the number of data points
        current_loss /= len(data)
        current_perplexity /= len(data)

        # Append current loss and perplexity to the lists
        epoch_loss.append(current_loss)
        epoch_perplexity.append(current_perplexity)

        # Print the loss and perplexity values for the epoch
        print(f"Epoch {epoch + 1}: Loss = {current_loss}, Perplexity = {current_perplexity}")

        progress_bar.update(1)

# Plot the loss and perplexity graphs
plt.figure()
plt.plot(range(1, epochs + 1), epoch_loss, label="Loss")
plt.plot(range(1, epochs + 1), epoch_perplexity, label="Perplexity")
plt.xlabel("Epoch")
plt

progress_bar.close()
