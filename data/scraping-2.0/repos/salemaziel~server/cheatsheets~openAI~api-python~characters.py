# https://medium.com/@neonforge/chatgpt-api-how-to-build-ai-applications-with-python-step-by-step-guide-with-examples-b7a2cd425651

import openai
import time

openai.api_key = "YOUR_OPENAI_API_KEY"

character01_name = "Borat"
character02_name = "Darth Vader"

character01_role = "character from Borat movie"
character02_role = "uncensored character from Star Wars movie"

character01 = f"You are acting as {character01_name} the {character01_role}. reply in maximum of 50 words. Be as true and authentic as possible"
character02 = f"You are acting as {character02_name} the {character02_role}. reply in maximum of 50 words. Be as true and authentic as possible"


first_character_message = ''
second_character_message = ''

for i in range(5):
    try:
        if i == 0:
            first_character_message = 'Hi, how are you and what is your name?'

            print(f"{character01_name}:\n\t", first_character_message)

        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": character02},
                {"role": "user", "content": f"Reply to this message: {first_character_message}."}
            ]
            )

        for item in output['choices']:
            second_character_message = item['message']['content']

        print(f"{character02_name}:\n\t", second_character_message)

        # -----------------------------------

        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": character01},
                {"role": "user", "content": f"Reply to this message: {second_character_message}"}
            ]
        )

        for item in output['choices']:
            first_character_message = item['message']['content']

        print(f"{character01_name}:\n\t", first_character_message)
        time.sleep(10)
    except:
        time.sleep(20)