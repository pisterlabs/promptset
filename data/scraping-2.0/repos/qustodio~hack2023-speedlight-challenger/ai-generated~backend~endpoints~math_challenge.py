import openai
import json
from flask import request
from flask_restful import Resource

from settings import CHAT_GPT_API_KEY

DIFFICULTY_PROMPTS = {
    'easy': "This is an easy math problem. ",
    'medium': "This is a medium difficulty math problem. ",
    'hard': "This is a challenging math problem. " 
}

class MathChallenge(Resource):
    def post(self):
        """Receive a math challenge and return the solution from ChatGPT."""
        challenge = request.get_json()['challenge']
        difficulty = request.get_json()['difficulty']
        challenge = challenge[:250]
        challenge = self.configure_prompt(challenge, difficulty)
        openai.api_key = CHAT_GPT_API_KEY
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": challenge}]
        )
        response = json.loads(str(chat_completion))
        
        # Return answer from endpoint
        return {'problem': response['choices'][0]['message']['content']}

    def configure_prompt(self, challenge, difficulty):
        if difficulty not in DIFFICULTY_PROMPTS:
            difficulty = 'easy'
        intial_prompt = DIFFICULTY_PROMPTS[difficulty] + challenge
        final_prompt = intial_prompt + '. Don\t show the solution. Answer in no more than 50 characters.'
        return final_prompt