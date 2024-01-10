import os
import time
from openai import OpenAI

# Start the timer
start_time = time.time()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Create a random message prompt
prompt = "Write a creative and random message about the day in the life of a space-traveling cat."

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-4",  # Assuming GPT-4 is available and this is the correct identifier
    max_tokens=50  # Limiting the response to 50 tokens
)

# Extract and print the response message
response_message = chat_completion.choices[0].message.content
print(response_message)

# Stop the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
