import openai
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv('OPENAI_API_KEY')


def prompt(uv_index):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Du är en rolig kompis som gillar att lägga kommentarer eller skämt."
            },
            {
                "role": "user",
                "content": f"Skriv en kort rolig kommentar utan citattecken om att det är UV-index {uv_index} idag. "
                           f"Den ska vara en mening lång. Låtsas vara en ung tjej eller kille. UV-index över 0-1 är "
                           f"väldigt lågt. UV-index 2 är lågt. UV-index 3 är normalt. UV-index 4-5 är lite högt. "
                           f"UV-index 6 och över är högt. Idag är det UV-index {uv_index}."
            }
        ],
        temperature=1.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(response)
    formatted_response = response['choices'][0]['message']['content'].strip()
    print(formatted_response)

    return formatted_response
