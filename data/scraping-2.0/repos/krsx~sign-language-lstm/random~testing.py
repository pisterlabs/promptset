import openai

# Replace 'your-api-key' with your actual API key
api_key = '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'
openai.api_key = api_key

# Your array of words
input_words = ["The", "quick", "brown", "fox"]

# Join the words into a single string
input_text = ' '.join(input_words)

# Generate a sentence
response = openai.Completion.create(
    engine="text-davinci-002",  # You can use other engines as well
    prompt=input_text + " is",
    max_tokens=50,  # Adjust this value to control the length of the generated sentence
    n=1  # Number of responses to generate
)

# Extract the generated sentence from the response
generated_sentence = response.choices[0].text.strip()

print(generated_sentence)

