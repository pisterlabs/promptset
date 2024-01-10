import openai
import os

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def calculate_perplexity(model, sentence):
    # Define the prompt with the sentence to evaluate

    # Define the completion parameters
    completion_parameters = {
        "model": model,
        "prompt": sentence,
        "max_tokens": 0,
        "logprobs": 0,
        "echo": True
    }

    # Call the OpenAI API to generate the completion
    response = openai.Completion.create(**completion_parameters)

    print(response)
    # Extract the perplexity from the API response
    choices = response['choices'][0]
    token_logprobs = choices['logprobs']['token_logprobs']

    print(token_logprobs)
    l = sum(token_logprobs[1:]) / len(token_logprobs[1:])
    perplexity = 2 ** (-l)

    return perplexity

# Example usage
sentence = "I love davinci"
model = "davinci"
perplexity = calculate_perplexity(model, sentence)
if perplexity is not None:
    print(f"The perplexity of the sentence is: {perplexity:.2f}")
else:
    print("Perplexity calculation failed.")