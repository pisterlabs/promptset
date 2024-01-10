from typing import Union, Tuple, Dict
import openai
import tiktoken
import os
from datetime import datetime
import warnings


def MessageTooLong(message_length: int, max_length: int = 8192):
    warnings.warn(f'Message length is {message_length} tokens, which is greater than the maximum allowed length of '
                  f'{max_length} tokens. Please shorten your message.')


class GPT:
    def __init__(self,
                 api_key: str,
                 model: str = 'gpt-4',
                 fast_model: str = 'gpt-3.5-turbo',
                 model_full_max_tokens: int = 8192,
                 max_token_answer: int = 2048,
                 anamnesis_length: int = 2048):

        self.api_key = api_key
        self.model = model
        self.fast_model = fast_model
        self.max_token_per_message = model_full_max_tokens
        self.max_token_answer = max_token_answer
        self.anamnesis_length = anamnesis_length

        self.base_max_token_query = self.max_token_per_message - self.max_token_answer  # Base we can send to the API

        self.MessagesHandler = MessagesHandler()

        self.GPT_model = openai.ChatCompletion(model=self.model)
        self.GPT_fast_model = openai.ChatCompletion(model=self.fast_model)

        self.ENCODER = tiktoken.encoding_for_model('text-davinci-003')

        self.prompt_paths = 'prompts'
        self.assistant_path = os.path.join(self.prompt_paths, 'assistant')
        self.system_path = os.path.join(self.prompt_paths, 'system')
        self.prebuilt_path = os.path.join(self.prompt_paths, 'prebuilt')
        self.evidence_based_path = os.path.join(self.prompt_paths, 'evidence_based_data')

        openai.api_key = self.api_key

        self.anamnesis = ''
        self.demand = ''

        self.last_model_used = None

    @staticmethod
    def get_content_api_query(content: dict) -> str:

        """
        Get the content of the API request
        :param content:
        :return:
        """

        return content['choices'][0]['message']

    def get_token(self, text: str):

        """
        Get the amount of token in a text for model davinci-003
        :param text:
        :return:
        """

        return len(self.ENCODER.encode(text))

    def query_API(self,
                  messages: list[dict],
                  key_to_protect: str = 'CONTEXT',
                  only_content: bool = True,
                  fast_model: bool = False,
                  **kwargs) -> Union[dict, str] or str:

        """
        Queries the GPT API with the given messages and ensures that the total token count does not exceed the maximum
         limit.

        :param fast_model: Runs on GPT-3.5 if True, otherwise runs on GPT-4.
        :param only_content: Returns only the content of the API response if True, otherwise returns the full response.
        :param messages: A list of dictionaries, each containing a 'role' and 'content' key representing the role
         (system or user) and the content of the message, respectively.
        :param key_to_protect: A string representing the keyword in the content that should be protected from being
         removed when reducing the token count. Default value is 'CONTEXT'.
        :return: The API response if the query is successful, otherwise None.
        """

        # DEBUG

        content = ''.join(message['content'] + ' ' for message in messages)

        if len_content := self.get_token(content) > self.max_token_per_message:
            MessageTooLong(len_content)

        # get which message to pop out. Not system!
        for index_to_pop, message in enumerate(messages):
            if message['role'] != 'system' and key_to_protect in message['content']:
                break

        # Cut the content
        while len_content > self.max_token_per_message:

            first_msg = messages.pop(index_to_pop)
            len_content -= self.get_token(first_msg['content'])

        # Check if a max_token has been passed and override the self parameter
        if 'max_tokens' in kwargs:
            max_token_answer = kwargs['max_tokens']
            del kwargs['max_tokens']

        else:
            max_token_answer = self.max_token_answer

        # QUERY
        try:
            if fast_model:
                response = self.GPT_fast_model.create(messages=messages,
                                                      model='gpt-3.5-turbo',
                                                      max_tokens=max_token_answer,
                                                      **kwargs)
            else:
                response = self.GPT_model.create(messages=messages,
                                                 model='gpt-4',
                                                 max_tokens=max_token_answer,
                                                 **kwargs)

            self.last_model_used = 'gpt-3.5-turbo' if fast_model else 'gpt-4'

        except Exception as e:  # TODO: Proper error handling

            response = None

            return e

        if response is None:

            return 'Error'

        else:
            return response['choices'][0]['message']['content'] if only_content else response

    def summarize(self, content: str,
                  target_tokens: int = 2048,
                  max_loop: int = 10,
                  return_process: bool = False) -> Union[str, Tuple[str, Dict[str, Dict[str, int]]]]:

        """
        Summarizes the given content to fit within the specified target token count by repeatedly querying the GPT API.

        :param content: A string containing the content to be summarized.
        :param target_tokens: An integer representing the target token count for the summarized content. Default value
         is 2048.
        :param max_loop: An integer representing the maximum number of iterations to perform when summarizing the
         content. Default value is 10.
        :param return_process: A boolean indicating whether to return the summarization process information along
        with the summarized content. Default value is False.
        :return: If return_process is False, returns the summarized content as a string. If return_process is True,
         returns a tuple containing the summarized content and a dictionary with the summarization process details.
        """

        if self.get_token(content) <= self.base_max_token_query:
            MessageTooLong(self.get_token(content), max_length=self.base_max_token_query)

        process_content = content[:self.base_max_token_query]

        msg_handler = MessagesHandler()

        reduced_data = {}
        i_loop = 0
        while (
            len_content := self.get_token(process_content) > target_tokens
        ) and i_loop <= max_loop:

            msg = [msg_handler.create_message(role='user',
                                              message=f'Please make the following content shorter: {content}',
                                              time=False,
                                              update_conv=False)]

            process_content = self.get_content_api_query(self.query_API(msg, key_to_protect='Please'))

            new_len_content = self.get_token(process_content)

            reduced_data[f'Loop {i_loop}'] = {'Query': len_content,
                                              'Answer': new_len_content,
                                              'Difference': len_content - new_len_content}

            len_content = new_len_content

            i_loop += 1

        return (process_content, reduced_data) if return_process else process_content

    def paraphrase(self, content: str, context: str = None) -> str:

        if context:
            prompt = f'CONTEXT:\n{context}\n' \
                     f'Please paraphrase:\n' \
                     f'CONTENT:\n{content}\n' \
                     f'Only answer with one paraphrase, based on the content given previously, and absolutely ' \
                     f'nothing else.'
        else:
            prompt = f'Please paraphrase:\n' \
                     f'{content}\n' \
                     f'Only answer with one paraphrase to the following sentence and absolutely nothing else.'

        msg_handler = MessagesHandler()

        msg = [msg_handler.create_message(role='user',
                                          message=prompt,
                                          time=False,
                                          update_conv=False)]

        return self.query_API(msg, key_to_protect='Please', only_content=True)

    @staticmethod
    def replace_conten_lists(tags: list[str], contents: list[str], replacements: list[str]) -> list[str]:

        assert len(tags) == len(contents) == len(replacements)

        return [
            content.replace(tag, replacement)
            for tag, content, replacement in zip(tags, contents, replacements)
        ]

    @staticmethod
    def chain_replacement(tags: list[str], replacements: list[str], content: str) -> str:

        assert len(tags) == len(replacements)

        for tag, replacement in zip(tags, replacements):
            content = content.replace(tag, replacement)

        return content


class MessagesHandler:
    def __init__(self):
        self.time = TimeHandling()

        self.ENCODER = tiktoken.encoding_for_model('text-davinci-003')

        self.conversation = []

    def create_message(self, role: str,
                       message: str,
                       time: bool = True,
                       full_time: bool = False,
                       command: str = 'not_given',
                       update_conv: bool = True):

        assert role in {
            'assistant',
            'system',
            'user',
        }, 'Role must be either "assistant", "system" or "user"'

        return_message = {
            'role': role,
            'content': message,
        }

        if time:
            return_message['time'] = self.time.get_full_time(update=True)

        if command != 'not_given':
            return_message['command'] = command

        if update_conv:
            self.conversation.append(return_message)

        return return_message

    def chat_to_GPT(self, chat: list[dict[str, str]]) -> list[dict[str, str]]:

        """
        Returns self conversation as a ChatCompletion valid message argument
        """

        return [
            {'role': message['role'], 'content': message['content']}
            for message in chat
        ] if chat else [
            {'role': message['role'], 'content': message['content']}
            for message in self.conversation
        ]

    def get_last_messages(self, token_limit: int = 2048, n_messages: int = 0):

        """
        Warning: Token limit ovverrides n_messages.

        :param token_limit:
        :param n_messages:
        :return:
        """

        if token_limit and n_messages:
            warnings.warn(f'Token limit ovverrides n_messages.'
                          f' Messages returned will be defined by token limit of {token_limit}')

        token = bool(token_limit)

        messages = []

        if token:

            token_count = 0

            for message in reversed(self.conversation):

                token_count += self.get_token(message['content'])

                if token_count >= token_limit:
                    break

                messages.append(message)

        else:

            for message in reversed(self.conversation):
                if len(messages) < n_messages:
                    messages.append(message)
                else:
                    break

        return reversed(messages)  # Return conversation in chronological order with full meta-data

    def get_token(self, content: str) -> int:

        return len(self.ENCODER.encode(content))

    def __len__(self):
        return len(self.conversation)


class TimeHandling:
    def __init__(self):
        self.now = datetime.now()

    def get_now(self):
        self.now = datetime.now()
        return self.now

    def get_full_time(self, update: bool = True):

        if update:
            self.now = datetime.now()

        return self.now.strftime('%c')

    def get_hour_minute(self, update: bool = True):

        if update:
            self.now = datetime.now()

        return self.now.strftime('%H:%M')

