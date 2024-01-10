import os
from core.utils import parse_reply_reflection
import openai

# Interface between LLM and character response generation
class LLM():

    def __init__(self):
        # Initialize the LLM details here.
        pass

    def get_response(self, prompt) -> tuple[str, str]:
        # Return (reply, reflection) given the prompt.
        pass


class GPT3(LLM):

    def __init__(self, debug=False):
        self.debug = debug
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # Get the raw response from GPT-3 with prompt using OpenAI API
    def get_raw_response(self, prompt):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.45,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response
    
    # Returns (reply, reflection) from the given prompt.
    def get_response(self, prompt):
        raw_response = self.get_raw_response(prompt)
        response_str = raw_response['choices'][0]['text']
        reply, reflection = parse_reply_reflection(response_str, debug=self.debug)
        return reply, reflection