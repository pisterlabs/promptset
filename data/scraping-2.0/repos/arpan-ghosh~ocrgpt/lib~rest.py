import os
import openai

from lib.utility import debug_print

DEFAULT_CREATION_CONTEXT = {
    "AI_MODEL" : "gpt-3.5-turbo",
    "TEMPERATURE" : 0,
    "MAX_TOKENS" : 100,
    "TOP_P" : 1,
    "FREQUENCE_PEANALTY" : 0,
    "PRESENCE_PENALTY" : 0,
    "STOP_TOKEN" : ["\n"]
}

class OpenAIRestLib():
    openai.api_key = ""
    
    def __init__(
            self,
            api_key,
            model=DEFAULT_CREATION_CONTEXT["AI_MODEL"],
            temperature=DEFAULT_CREATION_CONTEXT["TEMPERATURE"],
            max_tokens=DEFAULT_CREATION_CONTEXT["MAX_TOKENS"],
            top_p=DEFAULT_CREATION_CONTEXT["TOP_P"],
            frequence_penalty=None,
            presence_penalty=None,
            stop = None
            ):
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequence_penalty = frequence_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def list_gpt_models():
        try:
            response = openai.Model.list()
        except openai.error.APIError as e:
            debug_print("OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            debug_print("Failed to connect to OpenAI API: {e}")
            pass
        except openai.error.RateLimitError as e:
            debug_print("OpenAI API request exceeded rate limit: {e}")
            pass

        return response

    def create_chat_completition():
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
            )
        except openai.error.APIError as e:
            debug_print("OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            debug_print("Failed to connect to OpenAI API: {e}")
            pass
        except openai.error.RateLimitError as e:
            debug_print("OpenAI API request exceeded rate limit: {e}")
            pass

        debug_print(completion.choices[0].message.content)

        return completion.choices

    def ask_chatgpt(question):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=question,
                temperature=0,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n"]
            )
        except openai.error.APIError as e:
            debug_print("OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            debug_print("Failed to connect to OpenAI API: {e}")
            pass
        except openai.error.RateLimitError as e:
            debug_print("OpenAI API request exceeded rate limit: {e}")
            pass      

        debug_print(response)

        return response
