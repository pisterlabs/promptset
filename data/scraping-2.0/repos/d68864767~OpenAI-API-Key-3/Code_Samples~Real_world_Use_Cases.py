```python
# Real-world Use Cases of OpenAI API

# This script demonstrates some of the real-world use cases of the OpenAI API.
# It includes examples of content generation, chatbots in customer service, and AI-powered recommender systems.

import openai

# Ensure you have set the OpenAI API key
openai.api_key = 'your-api-key'

# Content Generation
def content_generation(prompt):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=prompt,
      temperature=0.5,
      max_tokens=100
    )
    return response.choices[0].text.strip()

# Chatbots in Customer Service
def customer_service_chatbot(message):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=message,
      temperature=0.5,
      max_tokens=100
    )
    return response.choices[0].text.strip()

# AI-powered Recommender Systems
def recommender_system(user_preferences):
    # This is a simplified example. In a real-world scenario, the model would need to be trained on a dataset of items.
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=user_preferences,
      temperature=0.5,
      max_tokens=100
    )
    return response.choices[0].text.strip()

# Test the functions
print(content_generation("Write a blog post about the future of AI."))
print(customer_service_chatbot("I have a problem with my order."))
print(recommender_system("I like science fiction movies and rock music."))
```
