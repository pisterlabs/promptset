import http.client
import os
import openai
import abc
import time
import sys
import tiktoken

chatgpt = 'gpt-3.5-turbo-0301'

def gpt(
    prompt,
    model=chatgpt,
    n=1,
    max_tokens=200,
    temperature=0,
    include_prompt=False,
    file=None
):
    """
    Generate text using gpt3, given a prompt.

    :param prompt: prompt input that model will append to via generation
    :param model: model to use-- ada is small, davinci is large
    :param n: number of generations to make using the prompt
    :param max_tokens: maximum number of tokens (prompt+generation) cutoff
    :param temperature: parameter controlling model output variance
    :param include_prompt: whether to include the prompt in outputs
    :param file: string name of file to append to
    :return: list of strings representing outputs (generations wrapped in square braces [])
    """
    if model == chatgpt:
        initial_time = time.perf_counter()
        while True:
            try:
                completions = openai.ChatCompletion.create(
                    model=model,
                    messages=[dict(role='user', content=prompt)],
                    temperature=temperature
                )
                break
            except (openai.error.RateLimitError, openai.error.Timeout, http.client.RemoteDisconnected, openai.error.APIError) as e:
                time_of_error = time.perf_counter()
                time_delta = time_of_error - initial_time
                initial_time = time_of_error
                print(f'Calling GPT resulted in an error after {time_delta:.2f}s. Error: {type(e)}. Retrying...', file=sys.stderr)
        outputs = [choice.message.content for choice in completions.choices]
        used_tokens = completions.usage.total_tokens
        display_outputs = [
            prompt + f'\n\n<{model}>[{output}]'
            for output in outputs
        ]
    else:
        while True:
            try:
                completions = openai.Completion.create(
                    engine=model,
                    temperature=temperature,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    n=n
                )
                break
            except (openai.error.RateLimitError, openai.error.Timeout, http.client.RemoteDisconnected) as e:
                time_of_error = time.perf_counter()
                time_delta = time_of_error - initial_time
                initial_time = time_of_error
                print(f'Calling GPT resulted in an error after {time_delta:.2f}s. Error: {type(e)}. Retrying...', file=sys.stderr)
        outputs = [choice.text for choice in completions.choices]
        display_outputs = [
            prompt + f'  <{model}>[{output}]'
            for output in outputs
        ]
    separator = '\n' +'_' * 80 + '\n\n'
    display_output = ''.join([o+separator for o in display_outputs])
    if isinstance(file, str):
        with open(file, 'a') as f:
            f.write(display_output)
    elif hasattr(file, 'write'):
        file.write(display_output)
    elif callable(file):
        file(display_output)
    if include_prompt:
        outputs = [prompt+output for output in outputs]
    if len(outputs) >= 2:
        return outputs, used_tokens
    else:
        return outputs[0], used_tokens

default = object()

class Prompt(abc.ABC):

    model = staticmethod(gpt)
    parse = None

    def __init__(self, *values, model=None, parse=None):
        self.prompt = self.template.format(*values)
        if model:
            self.model = model
        if parse:
            self.parse = parse

    @property
    @abc.abstractmethod
    def template(self): ...

    def generate(self, model=default, parser=default, logfile=None):
        model = self.model if model is default else model
        generated, used_tokens = model(self.prompt)
        if parser is default and self.parse:
            output = self.parse(generated)
        elif callable(parser):
            output = parser(generated)  # noqa
        else:
            output = None
        if logfile:
            logfile = logfile.replace('{}', self.__class__.__name__)
            with open(logfile, 'a+') as f:
                f.write('\n'.join((
                    f"{'_'*80}",
                    f"\n{self.__class__.__name__}: {self.__class__.__doc__}",
                    f"\n{self.prompt}<{model.__name__}>[{generated}]",
                    f"\n{output}" if output else ''
                )))
        return generated, output, used_tokens

    def record_duration(self, logfile, duration):
        logfile = logfile.replace('{}', self.__class__.__name__)
        with open(logfile, 'a+') as f:
            f.write(f"\nElapsed: {int(duration)} s\n")

def num_tokens_from_messages(inputs, output, model="gpt-3.5-turbo-0301"):
    """
    Returns the number of tokens used by a list of messages.
    Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in inputs:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3
    num_tokens += len(encoding.encode(output))
    return num_tokens
