import os
import re
import sys

import requests
import simplejson

from typing import List, Any, Dict

from .configurationhelpers import get_enable_openai, get_openai_api_key
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, ChatMessage, BaseMessage, LLMResult
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .loghelpers import LOG
from .jsonhelpers import load_from_json_file, save_to_json_file
from .self_hosted_LLM import SelfHostedLLM
from helpers.websockethelpers import broadcast_message, get_broadcast_channel, get_broadcast_sender

CLIENTS = {}


class BaseGeneration:
    def __init__(self, content: str):
        self.content = content

    def to_json(self) -> dict[str, str]:
        return {'content': self.content}


class TextGeneration(BaseGeneration):
    def __init__(self, content: str):
        super().__init__(content)

    def to_json(self) -> dict[str, str]:
        return {'content': self.content, 'type': 'text'}


class CodeGeneration(BaseGeneration):
    def __init__(self, content: str, language: str):
        super().__init__(content)
        self.language = language

    def to_json(self) -> dict[str, str]:
        return {'content': self.content, 'language': self.language, 'type': 'code'}


def parse_generation(input_string: str) -> list[dict[str, str]]:
    pattern = r"(?s)(```(?P<language>\w+)?\n(?P<code>.*?)```)|(?P<text>.*?(?=```|\Z))"
    matches = re.finditer(pattern, input_string)
    results = []
    for match in matches:
        if match.group('code'):
            results.append(CodeGeneration(match.group('code'), match.group('language')).to_json())
        elif match.group('text') is not None and match.group('text').strip():
            results.append(TextGeneration(match.group('text').strip()).to_json())
    return results


def get_llm(model_name: str = 'self-hosted', temperature: float = 0.0):
    global CLIENTS

    if model_name in CLIENTS:
        llm = CLIENTS[model_name]
        llm.model_name = model_name
        llm.temperature = temperature
        return llm

    if model_name.startswith('self-hosted'):
        models_file = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), 'configuration', 'LLMs.json')
        self_hosted_models = load_from_json_file(filename=models_file) if os.path.exists(models_file) else {}

        model_names = []
        for model in self_hosted_models:
            model_names.append(f'self-hosted:{model}')

        if model_name == 'self-hosted:MoE':
            LOG.info(f'Initializing {model_name} LLM with default settings')
            llm = SelfHostedLLM(mixture_of_experts=True)

        elif model_name in model_names:
            host = self_hosted_models[model_name.split(':')[1]]['host']
            port = self_hosted_models[model_name.split(':')[1]]['port']
            LOG.info(f'Initializing {model_name} LLM at {host}:{port}')
            llm = SelfHostedLLM(host=host, port=port, mixture_of_experts=False)
        else:
            LOG.info(f'Initializing {model_name} LLM with default settings')
            llm = SelfHostedLLM(mixture_of_experts=False)

        CLIENTS[model_name] = llm
        return llm

    if get_enable_openai() is True:
        if model_name == 'text-davinci-003':
            llm = OpenAI(model_name=model_name, temperature=temperature, openai_api_key=get_openai_api_key(), request_timeout=300)
        else:
            llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=get_openai_api_key(), request_timeout=300, streaming=True, callbacks=[CustomStreamingCallbackHandler()])

    else:
        raise Exception("OpenAI is not enabled")

    CLIENTS[model_name] = llm

    return llm


def get_role(message: BaseMessage):
    if isinstance(message, HumanMessage):
        return 'user: '
    elif isinstance(message, AIMessage):
        return 'assistant: '
    elif isinstance(message, SystemMessage):
        return 'system: '
    elif isinstance(message, ChatMessage):
        return message.role
    else:
        raise Exception("Unknown message type")


def comparison_prompt(messages: List[BaseMessage], generations: List[LLMResult]):
    original_prompt = ''
    for message in messages:
        original_prompt += message.content

    all_generations = ''
    for i, generation in enumerate(generations):
        all_generations += f'{i}: {generation.generations[0][0].text}\n'

    comparison_prompt = f"""Your task is to compare multiple generations from a LLM model to each other and choose the one that satisfies the original prompt the best.

## Original prompt for the LLM
{original_prompt}


## Generations to be compared
{all_generations}


## Instructions
Your answer should be formatted as a markdown code block containing a valid json object with the key 'best_n'.
The value of 'best_n' should be the index number of the best generation from the list of generations (index starts at 0).
for example:
```json
{{
  "best_n": 2
}}
```

Please respond with only the json object inside a markdown code block, and nothing else.
## Answer

"""

    return comparison_prompt


class LLM(object):
    llm = None
    model_name: str = None
    temperature: float = 0.0

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = get_llm(model_name, temperature)

    def generate(self, messages: List[BaseMessage], stop=None):
        kwargs = {'temperature': self.temperature}
        if self.model_name == 'text-davinci-003':
            prompts = []
            prompt = ''
            for message in messages:
                prompt += message.content + '\n'

            prompts.append(prompt)
            return self.llm.generate(prompts, stop=stop, **kwargs)
        else:
            return self.llm.generate([messages], stop=stop, **kwargs)

    def run(self, messages: List[BaseMessage], stop=None, best_of: int = 1):
        """Run the LLM and return the completion text, the LLM output, and the generation info.
        Note: generation info is only available for text-davinci-003.
        llm_output = {'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}

        :param messages: List - a list of messages
        :param stop: List - stop sequences
        :param best_of: int - number of completions to generate and return the best of
        :return: Tuple - completion text, LLM output, generation info
        """
        LOG.info(f'Running LLM {self.model_name}')

        results = [self.generate(messages, stop=stop) for _ in range(best_of)]

        if best_of > 1:
            LOG.info('Choosing best generation')
            llm_result = results[self.choose_best_generation(messages, results)]
        else:
            llm_result = results[0]

        completion_text = llm_result.generations[0][0].text
        generation_info = llm_result.generations[0][0].generation_info
        llm_output = llm_result.llm_output

        return completion_text, llm_output, generation_info

    def choose_best_generation(self, messages: List[BaseMessage], generations: List[LLMResult]) -> int:

        if self.model_name == 'text-davinci-003':
            result = self.llm.generate(comparison_prompt(messages, generations))
        else:
            result = self.llm.generate([[HumanMessage(content=comparison_prompt(messages, generations))]])

        try:
            parsed = parse_generation(result.generations[0][0].text)
        except Exception as e:
            LOG.error(f"Unable to parse generation, defaulting to first generation, invalid JSON: {e}")
            return 0

        as_json = {}
        for generation in parsed:
            if isinstance(generation, CodeGeneration):
                try:
                    as_json = simplejson.loads(generation.content)
                except Exception as e:
                    LOG.info(f"Unable to parse generation section as json: {e}")
                else:
                    break

        best_n = int(as_json.get('best_n', 0))

        if best_n >= len(generations):
            LOG.error(f"best_n is greater than the number of generations, defaulting to first generation")
            return 0

        return best_n


class CustomStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    """Custom callback handler for streaming. Only works with LLMs that support streaming."""

    def __init__(self):
        self.full_completion = ""

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.full_completion = ""
        LOG.info('streaming started again')

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        LOG.info('streaming ended')
        self.full_completion = ""
        data = {'message': '', 'channel': get_broadcast_channel(), 'sender': get_broadcast_sender(), 'parts': []}
        broadcast_message(message=simplejson.dumps(data), channel=get_broadcast_channel())

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        token = token.replace('\r', '')
        self.full_completion += token
        data = {'message': self.full_completion.lstrip(), 'channel': get_broadcast_channel(), 'sender': get_broadcast_sender(), 'parts': parse_generation(self.full_completion.lstrip())}
        broadcast_message(message=simplejson.dumps(data), channel=get_broadcast_channel())
        sys.stdout.write(token)
        sys.stdout.flush()


def load_llms():
    llms_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'configuration', 'LLMs.json')
    llms_data = load_from_json_file(filename=llms_file) if os.path.exists(llms_file) else {}

    return llms_data


def get_llm_config(llm_name: str):
    llms_data = load_llms()
    return llms_data.get(llm_name, {})


def delete_llm(llm_name: str):
    llms_data = load_llms()
    if llm_name in llms_data:
        del llms_data[llm_name]
        save_to_json_file(data=llms_data, filename=os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'configuration', 'LLMs.json'))


def save_llm_config(llm_name: str, llm_config: dict):
    llms_data = load_llms()

    if llm_config['server_type'] == 'Oobabooga':
        if llm_config['model_name'] is None and llm_config['host'] is not None:
            url = f"{llm_config['host']}/v1/models"
            headers = {
                "Content-Type": "application/json"
            }
            try:
                response = requests.get(url=url, headers=headers)
            except Exception as e:
                LOG.error(f"Unable to connect to {url}: {e}")
                return

            if response.status_code == 200:
                models = response.json()['data']
                print(models)
                for model in models:
                    if model['id'] not in ['gpt-3.5-turbo', 'text-embedding-ada-002']:  # these come with the server, ignore them
                        llm_config['model_name'] = model['id']
                        break
                    else:
                        llm_config['model_name'] = 'Unknown'

    if llm_config['model_name'] is None:
        LOG.error('No model name provided and unable to determine model name from server')
        return

    if llm_config['description'] is None:
        if llm_name.startswith('chat'):
            llm_config['description'] = 'Optimized for chat'
        elif llm_name.startswith('code'):
            llm_config['description'] = 'Optimized for code'

    llms_data[llm_name] = llm_config

    # if host ends with a trailing /, remove it
    if llm_config['host'] is not None and llm_config['host'].endswith('/'):
        llm_config['host'] = llm_config['host'][:-1]

    save_to_json_file(data=llms_data, filename=os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'configuration', 'LLMs.json'))
