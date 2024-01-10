#./app/chatbot/messaging_manager.py
from app.chatbot.prompt_builder import PromptBuilder
import openai
import requests

class MessagingManager:
    def __init__(self, bot_config):
        self.bot_config = bot_config
        self.api_key = self.bot_config.get_api_key()

    def send_message(self, message, chatbot_id):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        payload = {
            'prompt': message,
            'temperature': self.bot_config.get_config('temperature'),
            'max_tokens': self.bot_config.get_config('max_tokens'),
            'top_p': self.bot_config.get_config('top_p'),
            'frequency_penalty': self.bot_config.get_config('frequency_penalty'),
            'presence_penalty': self.bot_config.get_config('presence_penalty'),
            'stop': self.bot_config.get_config('stop')
        }
        response = requests.post(
            f'https://api.openai.com/v1/chatbots/{chatbot_id}/messages',
            headers=headers,
            json=payload
        )
        response_data = response.json()
        if response.status_code != 200:
            error_message = response_data.get('error', {}).get('message', 'Unknown error')
            raise ValueError(f'Failed to send message to chatbot: {error_message}')
        return response_data['choices'][0]['text']