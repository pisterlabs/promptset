```python
# Import the OpenAI SDK
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Example of using the SDK for text generation
def generate_text(prompt, max_tokens=60):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()

# Test the function
if __name__ == "__main__":
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    print(generate_text(prompt))
```
