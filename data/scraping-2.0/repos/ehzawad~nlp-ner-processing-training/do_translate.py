import os
import openai
import asyncio

openai.api_key = ''

async def translate_to_arabic(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Translate the following English text to Arabic: '{text}'"}
        ],
        temperature=0,
        max_tokens=256
    )
    return response.choices[0].message['content']

async def main():
    file_path = '/home/ehz/Downloads/language.txt'
    translated_file_path = 'translated.txt'

    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Writing the translated content to a new file
    with open(translated_file_path, 'w') as file:
        for line in content.strip().split('\n'):
            key, value = line.split('=')
            english_text = value.strip(" ';")
            arabic_text = await translate_to_arabic(english_text)
            file.write(f"{key.strip()}: '{arabic_text}',\n")
            print(f"Translated {key.strip()}: {arabic_text}")

    print("Translation completed and written to", translated_file_path)

# Running the asynchronous main function
asyncio.run(main())
