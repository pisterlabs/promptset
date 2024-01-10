# This code is to convert the text on the website to a better and rephrased version of the same text with the same meaning.
!pip install openai
!pip install --upgrade openai
!pip install openai==0.28

import openai

# Set your OpenAI API key
openai.api_key = 'your api key'

# Define your input text that you want to rephrase
input_text = "A blog (short for “weblog”) is an online journal or informational website run by an individual, group, or corporation that offers regularly updated content (blog post) about a topic. It presents information in reverse chronological order and it's written in an informal or conversational style."

# Make a request to the OpenAI API for text completion
response = openai.Completion.create(
    model="text-davinci-003",  # Specify the GPT-3 model
    prompt=f"Rephrase the following text:\n\"{input_text}\"",
    max_tokens=100,  # Adjust max_tokens as needed
    temperature=0.7  # Adjust temperature as needed
)

# Extract the generated text from the response
output_text = response['choices'][0]['text']

# Print the rephrased text
print(output_text)
