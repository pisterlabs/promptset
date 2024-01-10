import openai

# Replace 'YOUR_API_KEY' with your actual API key from OpenAI
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key

# Function to generate content using GPT-3
def generate_content(prompt, max_tokens=100):
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use the appropriate engine (e.g., text-davinci-002)
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,  # Number of responses to generate
        stop=None,  # Optional stop sequence to end the content generation
        temperature=0.7,  # Controls the randomness of the output (adjust as needed)
    )
    return response.choices[0].text.strip()

# Example usage
prompt = "i want story about cat love dog with 100 line"
content = generate_content(prompt)
print(content)
print('-------------------------')
content = generate_content('complite this '+content)
print(content)
