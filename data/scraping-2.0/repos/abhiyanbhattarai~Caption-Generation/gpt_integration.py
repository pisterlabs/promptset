import json
from openai import OpenAI


# Load your API key from the config file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

client = OpenAI(api_key=config.get('openai_api_key'))

#  Function to generate a better caption using GPT-3
def generate_gpt_caption(generated_caption,sentiment):

    # Generate OpenAI API request
    system_message = "You are generating a instagram caption more than 15 words, include hashtags and emojis.Generate maximum 2 hashtags and maximum 2 emojis"
    user_message = f"{sentiment} {generated_caption}"

    # Call OpenAI API
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    # Get the generated caption from OpenAI API response
    caption = completion.choices[0].message

    return caption.content