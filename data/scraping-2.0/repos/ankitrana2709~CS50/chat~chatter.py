import openai
import os

# Set up the OpenAI API key
openai.api_key = "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"

# Set up the initial conversation prompt
conversation_prompt = "Hello, I'm a chatbot. Which article you want today?"

# Set up the API parameters
model_engine = "davinci"
max_tokens = 150

# Start the conversation loop
while True:
    # Get the user's message
    article = "Write an article about "
    user_message = input("You: ")

    # Set up the prompt for the API request
    prompt = f"{conversation_prompt}\n\nUser: {article + user_message}\nBot:"

    # Generate the bot's response using the API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens
    )

    # Get the bot's response from the API response
    bot_message = response.choices[0].text.strip()

    # Print the bot's response
    print("Bot:", bot_message)

