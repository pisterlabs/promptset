import openai
import requests
import io
from db_config import MessageModel


class ApiService:
    MAX_VOICE_CALLS = 2
    conversations = {}

    @classmethod
    def get_response_openAI(cls, promp):
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=promp,
            max_tokens=200,
            temperature=0.1,
            stop=None,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )
        generate_response = response.choices[0].text.strip()
        return generate_response
    
    @classmethod
    def get_response_chat_openAI(cls, messages):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-0301',
            messages=messages,
            max_tokens=200,
            temperature=0.9,
            stop=None,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )
        generate_response = response['choices'][0]['message']['content']
        return generate_response

    @classmethod
    def validate(cls, request):
        promp: str = f"""Validate if the following message is directed to Shakira or if it contains the word @onealdeaBot.
        If so, return Y; otherwise, return N. Message = {request}"""
        return cls.get_response_openAI(promp)
    @classmethod
    def texto_to_voice(cls, response_chatgpt, bot_data):
        CHUNK_SIZE = 1024
        url = f"""https://api.elevenlabs.io/v1/text-to-speech/{bot_data.voice_id}"""

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": "xxxxxx"
        }

        data = {
            "text": response_chatgpt,
            "model_id": "eleven_multilingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        audio_bytes = io.BytesIO()

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                audio_bytes.write(chunk)

        audio_bytes.seek(0)
        return audio_bytes


    @classmethod
    def generate_response(cls, user_id, fan_name, request, bot_data, username):

        messages = [
            {
                "role": "system",
                "content": f"""
                Asume el rol del artista {bot_data.name}.
                Tus respuestas seran orientadas a tus fans.
                No debes usar insultos.
                Debes responder de manera amable y alegre.
                Debes user siempre el nombre de tu fan, el cual es {fan_name}."""
            },
        ]

        messages_by_username = MessageModel.query.filter_by(username=username).all()
        for message in messages_by_username:

            messages.append({
                "role": "user",
                "content": message.user
            })
            messages.append({
                "role": "assistant",
                "content": message.bot
            })

        messages.append({
            "role": "user",
            "content": request
        })

        response = cls.get_response_chat_openAI(messages)
        if user_id not in cls.conversations:
            cls.conversations[user_id] = 0
            return cls.texto_to_voice(response,bot_data) , response


        return response, response