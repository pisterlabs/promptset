import openai
import time
from openai.error import RateLimitError

api_key = "your API Key"
openai.api_key = api_key

def generate_content(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=300,
            temperature=0,
            stop=None
        )
        return response.choices[0].text.strip()
    except RateLimitError as e:
        print("Rate limit exceeded. Waiting and retrying...")
        time.sleep(60)  # Wait for a minute and then retry
        return generate_content(prompt)

# Get user input for the prompt
article_prompt = input("Enter the prompt: ")

# Generate and print the content
generated_content = generate_content(article_prompt)
print(generated_content)
