import private_config

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser

import openai
import tiktoken
import requests


class AiDevsRestapiHelper:
    """Encapsulate details about calling API, authorization and others."""

    def __init__(self, task_name):
        """Get authorization token."""
        self.task_name = task_name
        js = {'apikey': private_config.API_KEY_SERVICE}
        response = requests.post(private_config.TOKEN_URL + task_name, json=js)
        assert response.status_code < 300, {'HTTP code': response.status_code, 'JSON': response.json()}
        self._authorization_token = response.json()['token']

    def __enter__(self):
        """Init the class to use with 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit function for with statement."""
        pass

    def get_task(self, verbose=True):
        """Get task details."""
        response = requests.get(private_config.TASK_URL + self._authorization_token)
        assert response.status_code < 300, {'HTTP code': response.status_code, 'JSON': response.json()}
        if verbose:
            print(response.json())
        return response.json()

    def ask_question(self, question):
        response = requests.post(private_config.TASK_URL + self._authorization_token, data={'question': question})
        assert response.status_code < 300, {'question': question, 'HTTP code': response.status_code,
                                            'JSON': response.json()}
        print(response.json())
        return response.json()

    def submit_answer(self, answer):
        """Submit answer."""
        js = {'answer': answer}
        print(js)
        response = requests.post(private_config.ANSWER_URL + self._authorization_token, json=js)
        assert response.status_code < 300, {'Answer': js, 'HTTP code': response.status_code, 'JSON': response.json()}
        print(response.json())
        return response.json()


class ModeratorValidationException(Exception):
    """Given text is not valid, and it's violating our policy(ies)."""

    err_text = 'Given text violates policy(ies)!'

    def __init__(self):
        """Default constructor."""
        super().__init__(self.err_text)

    def __init__(self, text, details):
        """Constructor with details."""
        super().__init__({'title': self.err_text, 'text': text, 'details': details})


class OpenAiModerator:
    """Classifies if text violates OpenAI's Content Policy."""

    def __init__(self):
        """Init self - get authorization token."""
        self.api_key = private_config.OPENAI_API_KEY

    def invoke(self, text):
        """Wrapper to allow langchain chain."""
        self.verify(text)

    def verify(self, text):
        """Verify if message pass moderation. If not, rise an exception."""
        openai.api_key = self.api_key
        result = openai.Moderation.create(input=text)
        openai.api_key = None
        flagged = result['results'][0]['flagged']
        details = result['results'][0]
        if flagged:
            raise ModeratorValidationException(text, details)


class OpenAiHelper:
    """Wrapper for openai library to easy usage."""
    model_35 = 'gpt-3.5-turbo'
    model_40 = 'gpt-4'

    def __init__(self, model=model_35):
        """Init self - get authorization token, set default model etc."""
        self.model = model
        self.api_key = private_config.OPENAI_API_KEY
        self.encoding = tiktoken.encoding_for_model(model)

    def __enter__(self, model=model_35):
        """Init the class to use with 'with' statement."""
        self.model = model
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit function for with statement."""
        pass

    def chat_completion(self, messages):
        """Put messages to chat completion api. Parse output."""
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        openai.api_key = None

        text = response['choices'][0]['message']['content']
        ct = response['usage']['completion_tokens']
        pt = response['usage']['prompt_tokens']
        tt = response['usage']['total_tokens']

        print(f'Tokens: CT={ct}, PT={pt}, TT={tt}')

        return text

    def get_token_count(self, message):
        """Return token count bsed on tiktoken library."""
        return tiktoken.get_encoding(message)

    def encode(self, message):
        """Return token count bsed on tiktoken library."""
        return self.encoding.encode(message)


class SimpleOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a simple str. It removes content= at start"""

    def parse(self, text: str):
        """Really simple."""
        return text


class LangChainHelper:
    """Wrapper for LangChain commands."""

    SYSTEM_TEMPLATE = '''
As a {role} who answers the questions ultra-concisely and nothing more.
Truthfully say "I don't know" when I cannot find an answer.
    '''
    HUMAN_TEMPLATE = '{text}'

    def __init__(self):
        """Init self - get authorization token, init llm, chat etc."""
        self.api_key = private_config.OPENAI_API_KEY
        self.llm = OpenAI(openai_api_key=self.api_key)
        self.chat = ChatOpenAI(openai_api_key=self.api_key)

    def __enter__(self):
        """Init the class to use with 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit function for with statement."""
        pass

    def predict_messages(self, message):
        """Return predicted message."""
        result = self.chat.predict_messages(message)
        result = SimpleOutputParser().invoke(result)
        return result

    def get_answer_as_role(self, role, message):
        """Return answer in as given role."""
        chat_prompt = ChatPromptTemplate.from_messages([
            ('system', self.SYSTEM_TEMPLATE),
            ('human', self.HUMAN_TEMPLATE),
        ])
        chain = chat_prompt | self.chat | SimpleOutputParser()
        return chain.invoke({'role': role, 'text': message})
