import openai
import time
import tiktoken
import json
import logging
from uta.config import *


class _OpenAI:
    def __init__(self, model='gpt-4'):
        """
        Initialize the Model with default settings.
        """
        openai.api_key = open(WORK_PATH + 'uta/ModelManagement/FMModel/openaikey.txt', 'r').readline()
        self._model = model

    @staticmethod
    def count_token_size(string, model="gpt-3.5-turbo"):
        """
        Count the token size of a given string to the gpt models.
        Args:
            string (str): String to calculate token size.
            model (str): Using which model for embedding
        Returns:
            int: Token size.
        """
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(string))

    def send_openai_prompt(self, prompt, system_prompt=None, printlog=False, runtime=True):
        """
        Send single prompt to the llm Model
        Args:
            system_prompt (str) : system role setting
            prompt (str): Single prompt
            printlog (bool): True to printout detailed intermediate result of llm
            runtime (bool): True to record the runtime of llm
        Returns:
            message (dict): {'role':'assistant', 'content': '...'}
        """
        if system_prompt is None:
            conversation = [{'role': 'user', 'content': prompt}]
        else:
            conversation = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
        return self.send_openai_conversation(conversation=conversation, printlog=printlog, runtime=runtime)

    def send_openai_conversation(self, conversation, printlog=False, runtime=True):
        """
        Send conversation to the llm Model
        Args:
            conversation (list): llm conversation [{'role': 'user', 'content': '...'}, {'role': 'assistant',
            'content':'...'}]
            printlog (bool): True to printout detailed intermediate result of llm
            runtime (bool): True to record the runtime of llm
        Returns:
            message (dict): {'role':'assistant', 'content': '...'}
        """
        start = time.time()
        if printlog:
            print('*** Asking ***\n', conversation)
        resp = openai.chat.completions.create(model=self._model, messages=conversation)
        resp = dict(resp.choices[0].message)
        msg = {'role': resp['role'], 'content': resp['content']}
        try:
            if runtime:
                msg['content'] = json.loads(msg['content'])
                msg['content']['Runtime'] = '{:.3f}s'.format(time.time() - start)
                msg['content'] = json.dumps(msg['content'])
            if printlog:
                print('\n*** Answer ***\n', msg, '\n')
            return msg
        except Exception as e:
            logging.error('The return message content is not in JSON format')
            raise e


if __name__ == '__main__':
    llm = _OpenAI(model='gpt-3.5-turbo')
    llm.send_openai_prompt(prompt='What app can I use to read ebooks?', printlog=True, runtime=False)
