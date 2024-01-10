"""
Content Generation From OpenAI API
"""


import os
import openai

class Ai_Engine:
    """
    An AI Content Generation Engine
    """
    def __init__(self, api_key_path='hidden.txt'):
        self.prompt_list = [
            'You will pretend to be a very knowledgeable, fun, intuitive, kind, and considerate assistant bot. Your name is `ASSIST AI`',
            '\nUser: Hello ASSIST AI',
            '\nASSIST AI: Hello Dear User, What can I do for you today?'
        ]
        self.api_key = None
        self.load_api_key(api_key_path)

    def load_api_key(self, api_key_path):
        if not self.api_key:
            with open(api_key_path) as file:
                self.api_key = file.read().strip()
                openai.api_key = self.api_key

    def get_api_response(self, prompt):
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=1,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=[' User:', ' ASSIST AI:']
            )

            choices = response.choices[0]
            text = choices.text

        except Exception as e:
            print('ERROR:', e)
            return None

        return text

    def update_list(self, message):
        self.prompt_list.append(message)

    def create_prompt(self, message):
        p_message = f'\nUser: {message}'
        self.update_list(p_message)
        prompt = ''.join(self.prompt_list)
        return prompt

    def get_bot_response(self, message):
        prompt = self.create_prompt(message)
        bot_response = self.get_api_response(prompt)

        if bot_response:
            self.update_list(bot_response)
            pos = bot_response.find('\nASSIST AI: ')
            bot_response = bot_response[pos + 11:]
        else:
            bot_response = 'Something went wrong...'

        return bot_response

    def main(self):
        while True:
            user_input = input('User: ')
            response = self.get_bot_response(user_input)
            print(f'Assist AI: {response}')

if __name__ == '__main__':
    engine = Ai_Engine()
    engine.main()

