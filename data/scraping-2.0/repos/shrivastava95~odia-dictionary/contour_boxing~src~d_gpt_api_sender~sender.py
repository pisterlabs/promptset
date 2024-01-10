import openai
from dotenv import load_dotenv
import os
from tqdm import tqdm
import html
from time import sleep

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_GAUTAM')

openai.api_key = OPENAI_API_KEY


separator = '|'
source_folder = f'contour_boxing/parsed_texts'
output_destination = f'contour_boxing/GPT_outputs'
if not os.path.exists(output_destination):
    os.mkdir(output_destination)

for text_path in tqdm(os.listdir(source_folder)):
    input_path = os.path.join(source_folder, text_path)
    output_path_check = text_path
    output_path = os.path.join(output_destination, text_path)
    try:
        if output_path_check not in os.listdir(output_destination):
            # text_path = 'page6_0.png.pdf.txt'
            with open(input_path, 'rb') as input_text:
                input_text_decoded = html.unescape(input_text.read().decode('utf8'))
                completion = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    temperature = 0.8,
                    max_tokens = 2400,
                    messages = [
                        {
                            "role": "system", 
                            "content": f"You will help me parse raw noisy OCR output into neat tables. The output should contain the proper Odia characters instead of escape sequences."
                        },
                        {
                            "role": "user", 
                            "content": """The table should have three columns:
                1. Word in English
                2. Part of speech it belongs to
                3. Odiya Translation of the word

                Here is the raw data. Format it into the table. For example, here are the first two rows of the table:

                a | Adjective&Article   | ଅରୋଟ୍‌
                accordingly |	Adverb  |  ଆଦିଙ୍କ୍‌ ଲେକେ"""
                        },
                        {
                            "role": "assistant",
                            "content": "Sure! Please provide me with the raw data, and I'll format it into a three-column table for you."
                        },
                        {
                            "role": "user",
                            "content": input_text_decoded
                        }
                    ]
                )

            gpt_text = html.unescape(completion.choices[0].message['content'])
            with open(f'{output_path}', 'w+b') as f:
                f.write(gpt_text.encode())
            # sleep(5)
            # break
    except Exception as e:
        print(f'error occured during creation of {output_path}:')
        print(e)
        continue
    # break