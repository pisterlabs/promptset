import openai
import random
import os

# Set up your OpenAI API credentials
# GPT-3.5
openai.api_key = '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'
# GPT-4
# openai.api_key = os.getenv('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')

# input N
number = input("Enter number of names: ")
N = int(number)

def generate_random_names(num_names):
    prompt = "Generate {} random great male full names:".format(num_names)
    response = openai.Completion.create(
        engine='text-davinci-002',
        # engine = 'gpt-4',
        # engine='text-davinci-003',
        prompt=prompt,
        max_tokens=30,  # Set the desired length of the generated name
        # n=num_names,  # Generate a single response
        stop=None,  # Let the model determine the stopping point
        temperature=0.7  # Controls the randomness of the generated output
    )
    
    names = response.choices  # Extract the generated name from the API response
    return [name.text.strip() for name in names]


# Generate N random names
random_names = generate_random_names(N) 

# Print the generated names
for name in random_names:
    print(name)
