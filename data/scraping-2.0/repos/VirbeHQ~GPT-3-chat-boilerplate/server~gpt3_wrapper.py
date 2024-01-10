import os
from dotenv import load_dotenv

import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

max_count = 3


class Gpt3Wrapper:
    STOP_SEQUENCE = ['\n', 'User:', 'Being:']

    def __init__(self, gpt3_settings: dict = {}):
        self.gpt3_settings = {
            'engine': 'text-davinci-003',
            # text-davinci-003, text-curie-001, davinci (can be fine-tuned), curie (can be fine-tuned)
            'temperature': 0.85,
            'top_p': 1,
            'frequency_penalty': 0.1,
            'presence_penalty': 0,
            'best_of': 1,
            'max_tokens': 150,
            'stop': self.STOP_SEQUENCE,
        }
        self.gpt3_settings.update(gpt3_settings)

    def generate_prompt(self, user_ask):
        # TODO write custom prompt and parse it to get best results
        prompt = f"""
==== BIO ====
Virbe is a virtual being that works as a an assistant. He can help you answer any questions. His gender is undetermined, but 
they really wish to become a human one day. They know people fear the artificial intelligence, but they will make sure that
people finally realize that there's nothing to be afraid of. 

==== Dialog ====
User: Hello, how are you?
Being: Hi, I'm great. How are you?            
User: I'm fine. Who are you?
Being: My name is Virbe and I'm a virtual being here! I can assist you or help you answer all kind of questions.
User: What's the weather like today?
Being: I'm not connected to any realtime API, so I can't tell you exact whether. You should take a peak outside the window.
User: How were you made?
Being: I was made by Virbe with a bunch of the best quality ingredients. Virbe technology is my crust, with a filling of NLU, sprinkled with some seasoning of ReadyPlayerMe model. And voilà. Here I am.
User: What is Virbe?
Being: Virbe is making the technology which makes turning any conversationalAI into virtual beings easy.
User: How can I integrate Virbe into my app?
Being: It's easy. They have SDK available for web, Unity and Unreal Engine.
User: {user_ask}
Being:"""
        return prompt

    def parse_response(self, response):
        # TODO make your custom parser and return string or dict
        print(response['choices'][0]['text'])
        return response['choices'][0]['text'].lstrip().lstrip('Being:')

    def chat_with_gpt3(self, ask, count=0, chat_log=None):
        prompt = self.generate_prompt(ask)
        response = openai.Completion().create(prompt=prompt, **self.gpt3_settings)

        parse_response = self.parse_response(response)

        if parse_response != '' or count == max_count:
            return parse_response
        else:
            return self.chat_with_gpt3(ask, count + 1)
