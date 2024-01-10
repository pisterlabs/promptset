import openai
import dotenv
import os

class Resumer:
    def __init__(self) -> None:
        openai.api_key = dotenv.get_key(".env","API_KEY")

    def resume_file(self, file_name):
        file_path = os.path.join("cap", file_name)
        with open(file_path, 'r', encoding="utf8") as file:
            text = file.read()

        slice_size = 10000

        for i in range(0, len(text), slice_size):
            slice_text = text[i:i+slice_size]
            print(f'Slice original: {i} | {slice_text}')

            output_path = os.path.join("output", file_name)
            self.resume_slice(slice_text, output_path, i)

    def resume_slice(self, slice_text, output_path, i):
        response_chat_gpt = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você resume textos em português de maneira objetiva"},
                    {"role": "user", "content": slice_text},
                ]
            )

        response = ''
        for choice in response_chat_gpt.choices:
            response += choice.message.content

        print(f'Slice resumida: {i} | {response}')

        with open(output_path, 'a') as file:
            file.write(response)
