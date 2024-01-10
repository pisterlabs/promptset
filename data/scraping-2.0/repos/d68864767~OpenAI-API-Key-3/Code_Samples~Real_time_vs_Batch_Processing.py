```python
# Code_Samples/Real_time_vs_Batch_Processing.py

import openai
import time

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Define a list of prompts for batch processing
prompts = [
    "Translate the following English text to French: '{}'",
    "Summarize the following text: '{}'",
    "What is the sentiment of the following text: '{}'"
]

# Define a list of texts to process
texts = [
    "Hello, world!",
    "In a far away kingdom, there lived a kind and just king. He was loved by all his subjects and ruled with wisdom and courage.",
    "I absolutely love this product! It has changed my life for the better and I can't imagine my daily routine without it."
]

# Real-time Processing
print("Real-time Processing Results:")
for prompt, text in zip(prompts, texts):
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt.format(text), max_tokens=60)
    print(response.choices[0].text.strip())
    time.sleep(1)  # To prevent hitting rate limits

# Batch Processing
print("\nBatch Processing Results:")
documents = [{'prompt': prompt.format(text), 'max_tokens': 60} for prompt, text in zip(prompts, texts)]
responses = openai.Completion.create(engine="text-davinci-002", n=3, documents=documents)
for response in responses:
    print(response.choices[0].text.strip())
```
