from .nlpbot import NlpBot
import openai

class Gpt3Bot(NlpBot):
    def __init__(self, api_key:str, model:str = 'text-davinci-002'):
        # This abstraction is leaky here. The abstraction suggests api_key is specific to this
        # bot but it is actually global as implemented here. This could be improved.
        openai.api_key = api_key
        self.model = model
        self.max_tokens = 256 # This limit is the sum of both input tokens and output tokens.
        self.temperature = 0.7 # 0-1. As we approach 0 behavior is deterministic and repetitive, 1 is more outlandish.
        self.top_p = 1 # 0-1. Diversity of options: 0.5 means half of all likelihood-weighted options are considered.
    
    def complete(self, prompt:str) -> str:
        response = openai.Completion.create(
          model = self.model,
          prompt = prompt,
          temperature = self.temperature,
          max_tokens = self.max_tokens,
          top_p = self.top_p,
          frequency_penalty=0,
          presence_penalty=0
        )
        response_text = response['choices'][0]['text']
        return response_text