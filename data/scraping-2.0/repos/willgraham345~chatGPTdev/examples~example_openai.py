import openai

# Your API key
api_key = "YOUR_API_KEY"

# Article content
article = """
This is the content of the article you want to summarize. It can be quite long and detailed.
You can include paragraphs or sections of the article here.
"""

# Compose a prompt for ChatGPT
prompt = f"Summarize the following article: {article}"

# Make a request to the ChatGPT API
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=50,  # Adjust the max_tokens for desired summary length
    api_key=api_key
)

# Extract and print the summary
summary = response.choices[0].text
print(summary)