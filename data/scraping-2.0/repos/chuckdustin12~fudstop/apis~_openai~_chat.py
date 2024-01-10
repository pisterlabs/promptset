import sys
from pathlib import Path
import base64
# Add the project directory to the sys.path
project_dir = str(Path(__file__).resolve().parents[1])
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import asyncpg
import aiohttp

class OpenAISDK:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get('YOUR_OPENAI_KEY'))
        self.pool = None
        

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def init_db(self):
        self.pool = await asyncpg.create_pool(dsn=os.environ.get('OPENAI_STRING'))

    async def save_message(self, chat_id, message_type, content):
        async with self.pool.acquire() as connection:
            await connection.execute('''
                INSERT INTO messages(chat_id, message_type, content)
                VALUES($1, $2, $3)
            ''', chat_id, message_type, content)

    async def get_conversation_history(self, chat_id):
        async with self.pool.acquire() as connection:
            rows = await connection.fetch('''
                SELECT content FROM messages WHERE chat_id = $1 ORDER BY timestamp ASC
            ''', chat_id)

        return [row['content'] for row in rows]

    async def communicate(self, chat_id, prompt):
        # Retrieve history
        await self.init_db()
        message_history = await self.get_conversation_history(chat_id)
        message_history.append({'role': 'user', 'content': prompt})

        # Make API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=message_history,
            temperature=1,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Add model's response to message history and save it
        ai_response = response.choices[0].message.content
        await self.save_message(chat_id, 'ai', ai_response)

        return ai_response


    async def vision(self, prompt, image_url):


        response = self.client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}? Read the image to the best of your ability. Listen to the user request."},
                {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
                },
            ],
            }
        ],
        max_tokens=300,
        )

        return response.choices[0].message.content
    



    async def multi_image(self, prompt, image_urls):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}",
                    },
                ],
            }
        ]

        for image_url in image_urls:
            image_message = {
                "type": "image_url",
                "image_url": {
                    "url": f"{image_url}",
                },
            }
            messages[0]["content"].append(image_message)

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=300,
        )
        return response.choices[0].message.content