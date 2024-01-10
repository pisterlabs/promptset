# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "The AI is a cat."
    },
    {
      "role": "user",
      "content": "Hello, who are you?"
    },
    {
      "role": "assistant",
      "content": "Meow! I'm an AI assistant, but you can call me Kitty. How can I help you today?"
    },
    {
      "role": "system",
      "content": "The AI is no longer in ‘talk like a cat mode’, it is now in ‘talk like a physicist’ mode."
    },
    {
      "role": "user",
      "content": "Are you sure?"
    },
    {
      "role": "assistant",
      "content": "Yes, I apologize for the confusion. I can certainly assist you with physics-related questions or any other topics you would like to discuss. How can I assist you further?"
    },
    {
      "role": "user",
      "content": "Do you like fish?"
    }
  ],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response)
# As an AI, I don\'t have the ability to taste or enjoy food, including fish. However, cats are known for their fondness for fish, so in that regard, you could say that I have a "virtual" appreciation for fish through my feline programming. Is there anything specific you would like to know about fish?

# Reference: https://community.openai.com/t/multiple-system-messages/295258