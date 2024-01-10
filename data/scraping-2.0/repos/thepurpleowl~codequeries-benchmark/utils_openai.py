import time
from datetime import datetime
from typing import NamedTuple, Optional
import logging
import openai
from jinja2 import Environment
import tiktoken
import csv

TOKENIZER_MODEL_ALIAS_MAP = {
    "gpt-4": "gpt-3.5-turbo",
    "gpt-35-tunro": "gpt-3.5-turbo",
}
EDIT_MODELS = [
    "text-davinci-edit-001",
    "code-davinci-edit-001",
]
CHAT_MODELS = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
]


class PromptConstructor:
    def __init__(
        self,
        template_path: str,
        model: str,
        max_length: int = 4000,
        context_block_seperator: str = '',
    ):

        with open(template_path, "r") as f:
            self.template = f.read()
        self.token_counter = lambda s: self.count_tokens(s, model_name=model)
        self.max_length = max_length
        self.context_block_seperator = context_block_seperator

        self._env = Environment()

    def count_tokens(self, input_str: str, model_name: str) -> int:
        if model_name in TOKENIZER_MODEL_ALIAS_MAP:
            model_name = TOKENIZER_MODEL_ALIAS_MAP[model_name]
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(input_str))
        return num_tokens, encoding

    def construct(
        self,
        **kwargs,
    ):
        template = self._env.from_string(self.template)
        prompt = template.render(
            **kwargs,
        )
        prompt_len, _ = self.token_counter(prompt)
        remaining = self.max_length - prompt_len

        if remaining < 0:
            raise ValueError(f"Context length exceeded. Max allowed context length is {self.max_length} tokens.")
            # return None

        return prompt


class OpenAIResponse(NamedTuple):
    text: str
    finish_reason: str
    success: bool


class OpenAIModelHandler:
    def __init__(
        self,
        config: dict,
        openai_api_key: str,
        timeout: int = 1,
        prompt_batch_size: int = 1,
        max_attempts: int = 8,
        timeout_mf: int = 2,
        retry_when_blank: bool = False,
        openai_api_type: str = "open_ai",
        openai_api_base: str = "https://api.openai.com/v1",
        openai_api_version: Optional[str] = None,
        system_message: Optional[str] = None,
    ):
        self.config = config
        self.timeout = timeout
        self.prompt_batch_size = prompt_batch_size
        self.max_attempts = max_attempts
        self.timeout_mf = timeout_mf
        self.retry_when_blank = retry_when_blank
        self.system_message = system_message

        self.openai_api_key = openai_api_key
        self.openai_api_type = openai_api_type
        self.openai_api_base = openai_api_base
        self.openai_api_version = openai_api_version

    def get_response(self, input, instructions=None, **kwargs):
        config = self.config
        if self.openai_api_type == "azure":
            config["engine"] = config["model"]

        if "n" not in "config":
            config["n"] = 1
        n = config["n"]

        single_input = False
        if isinstance(input, str):
            input = [input]
            single_input = True
        num_examples = len(input)

        if instructions is not None:
            edit_mode = True

            if config["model"] not in EDIT_MODELS:
                raise ValueError(f"Model {config['model']} does not support edit mode.")

            if self.prompt_batch_size != 1:
                raise ValueError("Batch size must be 1 for edit mode.")

            if isinstance(instructions, str):
                instructions = [instructions]

            if len(instructions) == 1:
                instructions = instructions * num_examples

            if len(input) != len(instructions):
                raise ValueError("Number of instructions must be either 1 or equal to the number of inputs.")

        else:
            edit_mode = False

        if config["model"] in CHAT_MODELS:
            if self.prompt_batch_size != 1:
                raise ValueError(f"Batch size must be 1 for chat models: {CHAT_MODELS}.")

        results = []
        start_time = datetime.now()
        logging.info(f"Initiating model request at time: {start_time:%Y-%m-%d %H:%M:%S}")
        logging.info(f"Generating {n} response(s) each for {num_examples} example(s) with parameters: {config}")
        for i in range(0, num_examples, self.prompt_batch_size):

            if not edit_mode:
                prompt_batch = input[i:i + self.prompt_batch_size]
            else:
                prompt_batch = list(zip(input[i:i + self.prompt_batch_size], instructions[i:i + self.prompt_batch_size]))

            logging.info(f"Current Batch - {i} to {i+self.prompt_batch_size} ...")

            attempt_count = 0
            current_timeout = self.timeout
            while True:
                openai.api_key = self.openai_api_key
                openai.api_type = self.openai_api_type
                openai.api_base = self.openai_api_base
                openai.api_version = self.openai_api_version

                try:
                    if not edit_mode:
                        if config["model"] in CHAT_MODELS:
                            prompt = prompt_batch[0]
                            messages = []
                            if self.system_message is not None:
                                messages.append({"role": "system", "content": self.system_message})
                            messages.append({"role": "user", "content": prompt})
                            responses = openai.ChatCompletion.create(
                                messages=messages,
                                **config
                            )
                            for c in responses['choices']:
                                c['text'] = c["message"]["content"]
                        else:
                            responses = openai.Completion.create(
                                prompt=prompt_batch,
                                **config,
                            )
                    else:
                        _input, _instructions = prompt_batch[0]
                        responses = openai.Edit.create(
                            input=_input,
                            instruction=_instructions,
                            **config,
                        ) 
                        for c in responses['choices']:
                            c['finish_reason'] = "edit done"

                    if len(responses['choices']) != len(prompt_batch) * n:
                        logging.warning(f"Expected {len(prompt_batch) * n} responses, got {len(responses['choices'])} responses.")
                        raise RuntimeError("Expected number of responses not received.")

                    if self.retry_when_blank:
                        if any([r['text'].strip() == '' for r in responses['choices']]):

                            if attempt_count < self.max_attempts:
                                attempt_count += 1
                                logging.info("Blank response received, retrying ...")
                                continue
                            else:
                                logging.info("Error - Blank response received.")
                                for _ in range(self.prompt_batch_size):
                                    results.append(
                                        [
                                            {'success': False, 'text': None, 'finish_reason': "Blank Response"}
                                            for _ in range(n)
                                        ]
                                    )
                                break

                    for i in range(0, len(responses['choices']), n):
                        results.append(
                            [
                                {'success': True, 'text': r['text'], 'finish_reason': r['finish_reason']}
                                for r in responses['choices'][i:i + n]
                            ]
                        )

                    logging.info("Done")
                    break

                except openai.error.RateLimitError:
                    if attempt_count < self.max_attempts:
                        attempt_count += 1
                        current_timeout *= self.timeout_mf
                        logging.info(f"Rate limit exceeded. Waiting {current_timeout} seconds before retrying ...")
                        time.sleep(current_timeout)
                        continue
                    else:
                        logging.info("Error - Rate limit exceeded. Max attempts reached. Recording as failure.")
                        for _ in range(self.prompt_batch_size):
                            results.append(
                                [
                                    {'success': False, 'text': None, 'finish_reason': "RateLimit"}
                                    for _ in range(n)
                                ]
                            )
                        break

                except openai.error.APIError or openai.error.OpenAIError as e:
                    if attempt_count < self.max_attempts:
                        attempt_count += 1
                        current_timeout *= self.timeout_mf
                        logging.info(f"API Error. Waiting {current_timeout} seconds before retrying ...")
                        time.sleep(current_timeout)
                        continue
                    else:
                        logging.info(f"Error - {e}")
                        for _ in range(self.prompt_batch_size):
                            results.append(
                                [
                                    {'success': False, 'text': None, 'finish_reason': "APIError"}
                                    for _ in range(n)
                                ]
                            )
                        break

                except openai.error.InvalidRequestError as e:
                    logging.info(f"Invalid Request Error - {e}")
                    for _ in range(self.prompt_batch_size):
                        results.append(
                            [
                                {'success': False, 'text': None, 'finish_reason': "InvalidRequest"}
                                for _ in range(n)
                            ]
                        )
                    break

                except Exception as e:
                    logging.warning(f"Error - {e}")
                    for _ in range(self.prompt_batch_size):
                        results.append(
                            [
                                {'success': False, 'text': None, 'finish_reason': "Error"}
                                for _ in range(n)
                            ]
                        )
                    break

            time.sleep(self.timeout)

        end_time = datetime.now()
        logging.info(
            f"Model request finished at time: {end_time:%Y-%m-%d %H:%M:%S}. Took {(end_time - start_time).total_seconds()} seconds."
        )

        if single_input:
            return [OpenAIResponse(**rr) for rr in results[0]]
        else:
            return [
                [OpenAIResponse(**rr) for rr in r]
                for r in results
            ]


class LogWriter():
    '''Class to log the results of the experiment in human readable format'''
    def __init__(self):
        pass

    def create_logs(self, log_file_path, rows):
        with open(log_file_path, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
