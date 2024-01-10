import os
import openai
from langchain.chat_models import ChatOpenAI
from llm.polymath.app.prompts.business_model_prompt import get_prompt_messages

class OpenAIChat:
    def __init__(self):
        openai.api_key = os.environ['OPENAI_API_KEY']
        # To control the randomness and creativity of the generated
        # text by an LLM, use temperature = 0.0
        self.chat = ChatOpenAI(temperature=0.7)

    def get_response(self, prompt, model='gpt-3.5-turbo', max_tokens=100):
        response = openai.Completion.create(
          engine=model,
          prompt=prompt,
          max_tokens=max_tokens
        )
        return response.choices[0].text.strip()


