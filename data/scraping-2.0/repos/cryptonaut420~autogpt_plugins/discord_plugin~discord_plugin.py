# Import the discord module
import discord
import os
import requests
import time
import openai
import tiktoken

def count_tokens(string: str, encoding_name: str = "gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Ask a question on discord, wait for a reply before proceeding
def _ask_on_discord(question):
    # Define the API base URL and headers
    api_base_url = "https://discord.com/api/v10"
    headers = {
        "Authorization": f"Bot {os.getenv('DISCORD_TOKEN')}",
        "Content-Type": "application/json"
    }

    # Load the channel ID
    channel_id = os.getenv('DISCORD_CHANNEL_ID')

    # Send the message
    message_url = f"{api_base_url}/channels/{channel_id}/messages"
    payload = {"content": str(question)}
    response = requests.post(message_url, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Failed to send the question. Status code: {response.status_code}"

    # Poll the API for new messages until a reply is received
    sent_message_id = response.json()["id"]
    while True:
        time.sleep(5)  # Wait for 5 seconds before checking for new messages

        # Get the latest messages in the channel
        response = requests.get(message_url, headers=headers, params={"after": sent_message_id})

        if response.status_code != 200:
            return f"Failed to get messages. Status code: {response.status_code}"

        messages = response.json()
        if len(messages) > 0:
            # Get the user's username and the reply content
            username = messages[0]["author"]["username"]
            reply = messages[0]["content"]

            # Return the reply with the user's @ username
            return f"@{username} answers: {reply}"



#Post on discord without expecting a reply
def _post_on_discord(message):
    # Load the webhook URL
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

    # Prepare the payload
    payload = {
        "content": str(message)
    }

    # Send the message using the webhook URL
    response = requests.post(webhook_url, json=payload)

    # Check if the request was successful
    if response.status_code == 204:
        return "Discord message posted!"
    else:
        return f"Failed to post the Discord message. Status code: {response.status_code}"

#check on discord channel messages
def _check_discord_channel():
    # Define the API base URL and headers
    api_base_url = "https://discord.com/api/v10"
    headers = {
        "Authorization": f"Bot {os.getenv('DISCORD_TOKEN')}",
        "Content-Type": "application/json"
    }

    # Load the channel ID and bot user ID
    channel_id = os.getenv('DISCORD_CHANNEL_ID')
    bot_user_id = os.getenv('DISCORD_BOT_USER_ID')

    # Get the latest messages in the channel
    message_url = f"{api_base_url}/channels/{channel_id}/messages"
    response = requests.get(message_url, headers=headers, params={"limit": 100})

    if response.status_code != 200:
        return f"Failed to get messages. Status code: {response.status_code}"

    messages = response.json()

    # Filter messages not created by the bot and combine them into a single string
    messages_text = "\n".join(
        [f"@{msg['author']['username']}: {msg['content']}" for msg in messages if msg['author']['id'] != bot_user_id][-25:][::-1]
    )

    # Check token count using tiktoken
    encoding_name = "gpt2"
    token_count = count_tokens(messages_text, encoding_name)

    # Summarize the content using OpenAI API if the token count exceeds 1000
    if token_count > 1000:
        # Load the OpenAI API key
        openai.api_key = os.getenv('OPENAI_API_KEY')

        # Create the prompt for summarization
        prompt = f"Please summarize the following text:\n\n{messages_text}"

        # Call the OpenAI API to generate a summary
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Extract the summary from the response
        summary = response.choices[0].text.strip()

        return summary
    else:
        return messages_text
