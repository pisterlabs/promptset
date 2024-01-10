import openai
import requests
from decouple import config

openai.api_key = config('OPENAI_API_KEY')
BOT_APP_ID = config('BOT_APP_ID')
    
def GPT(data, command_name, user_message, interaction_token):
    roles_content = determine_roles_content(command_name, data)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": roles_content},
            {"role": "user", "content": user_message},
        ],
        max_tokens=666,
    )

    print(response)
    # usage = response['usage']
    # + f"\n\nUsage: {usage}"
    # save for the show usage command
    
    message_content = response['choices'][0]['message']['content']
    
    # Truncate the response if it's too long
    if len(message_content) > 3000:
        message_content = "The response is too long. Due to Discord's character limit, the response has been truncated."

    # Edit the thinking message with the bot's response
    edit_url = f"https://discord.com/api/v8/webhooks/{BOT_APP_ID}/{interaction_token}/messages/@original"
    requests.patch(edit_url, json={"content": message_content})

    return {}  # No further data needs to be returned since we've already responded via editing

# Determine the content of the roles
def determine_roles_content(command_name, data):
    roles_content_map = {
        "chat": "",
        "chat_emo": "You will be provided with a message, and your task is to respond using emojis only.",
        "chat_multplechoice": "You will be provided with a multiple-choice problem, and your task is to only output the correct answer.",
        "chat_custom": data["options"][0]["value"]
    }
    return roles_content_map.get(command_name, "")