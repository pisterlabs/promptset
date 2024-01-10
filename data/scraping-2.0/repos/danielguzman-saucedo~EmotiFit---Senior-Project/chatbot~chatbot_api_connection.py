!pip install openai
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(
    api_key="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"  # Replace with your actual API key
)

# Create a chat completion using the OpenAI client
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Hello, chatbot!"
        }
    ],
    model="gpt-3.5-turbo",  # Specify the model you are using
)

response_message = chat_completion.choices[0].message.content

# Print the response from the chatbot
print(response_message)
