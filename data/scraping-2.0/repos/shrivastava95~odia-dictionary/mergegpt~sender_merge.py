import openai
from dotenv import load_dotenv
import os
from tqdm import tqdm
import html
from time import sleep

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY



source_folder = f'../odia-dict-repo-psm2/parsed_texts'           # a copy of the cloned repo with original psm settings
source_folder_2 = f'../odia-dict-repo/parsed_texts'             # a copy of the cloned repo with different psm settings
output_destination = f'mergegpt/GPT_outputs_merged'

for text_path in tqdm(os.listdir(f'{source_folder}')):
    input_path = f"{source_folder}/{text_path}"
    input_path_2 = f"{source_folder_2}/{text_path}"
    output_path_check = f'{text_path}'
    output_path = f'{output_destination}/{text_path}'
    try:
        if output_path_check not in os.listdir(output_destination):
            # text_path = 'page6_0.png.pdf.txt'
            with open(input_path_2, 'rb') as input_text_2:
                input_text_decoded_2 = html.unescape(input_text_2.read().decode('utf8'))
            with open(input_path, 'rb') as input_text:
                input_text_decoded = html.unescape(input_text.read().decode('utf8'))
            
            messages = [
                {
                    "role": "system", 
                    "content": "You parse raw noisy OCR output into neat tables."
                },
                {
                    "role": "user", 
                    "content": """The table should have three columns:
1. Word in English
2. Part of speech it belongs to
3. Odiya Translation of the word

I will give you two OCR outputs. Some entries hold erroneous values. For any given position, it may be such that only one of the outputs, or both, or none have the desired value. Your job is to merge the outputs in such a way that the errors are reduced.
Here is the raw output 1. Format it into the table. For example, here are the first two rows of the table:

a | Adjective&Article   | ଅରୋଟ୍‌
accordingly |	Adverb  |  ଆଦିଙ୍କ୍‌ ଲେକେ"""
                },
                {
                    "role": "assistant",
                    "content": "Sure! Please provide me with the raw output 1."
                },
                {
                    "role": "user",
                    "content": f"""{input_text_decoded}"""
                },
                {
                    "role": "assistant",
                    "content": "Please provide me with the raw output 2, and I'll format it into a three-column table for you after merging with output 2 while considering the removal of non-common errors in both."
                },
                {
                    "role": "user",
                    "content": f"""{input_text_decoded_2}"""
                }
            ]
            # counts number of tokens in the message sequence. should be less than 4096
            num_tokens = sum([len(dictionary['content']) for dictionary in messages])
            # print(num_tokens)
            # sleep(2)
            # break
            # continue

            completion = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                temperature = 0.8,
                max_tokens = 1200,
                messages = messages
            )

            gpt_text = html.unescape(completion.choices[0].message['content'])
            with open(f'{output_path}', 'w+b') as f:
                f.write(gpt_text.encode())
            sleep(5)
            # break
    except Exception as e:
        print(f'error occured during creation of {output_path}:')
        print(e)
        continue
    # break