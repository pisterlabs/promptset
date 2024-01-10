import openai
import os
import json
from transcripts_db import TranscriptsDB
from count_tokens import Counter
import requests

openai_api_key = os.getenv("OPENAI_API_KEY")
azure_gpt4_api_key = os.getenv("AZURE_GPT4_API_KEY")
gpt4_api_key = os.getenv("GPT4_API_KEY")

class Golem:

    def __init__(self, api_key, session_id, sys_prompt="", sys_prompt_prefix="", sys_prompt_suffix="", user_input_prefix="", user_input_suffix="", max_tokens=None, temperature=0.7, memory=False, is_stream=True, table_name="", column_name="", is_gpt4=None, is_azure_gpt4=False):
        self.__model = "gpt-4" if (is_azure_gpt4 or is_gpt4)  else "gpt-3.5-turbo"
        self.__openai_api_key = api_key
        self.__session_id = session_id
        self.__user_input_prefix = user_input_prefix
        self.__user_input_suffix = user_input_suffix
        self.__max_tokens = max_tokens
        self.__temperature = temperature
        self.__memory = memory
        self.__is_azure_gpt4 = is_azure_gpt4
        self.__init_sys_prompt = [
            {'role': 'system', 'content': sys_prompt_prefix + sys_prompt + sys_prompt_suffix}]
        self.__is_stream = is_stream
        self.__table_name = table_name
        self.__column_name = column_name

        if self.__memory:
            self.__transcripts_db = TranscriptsDB()
            with self.__transcripts_db as db:
                db.create_table(self.__table_name, self.__column_name)
                transcript_history = db.retrieve_data(
                    self.__table_name, self.__session_id, self.__column_name)
            self.__transcript_history = transcript_history if transcript_history else self.__init_sys_prompt
        else:
            self.__transcript_history = self.__init_sys_prompt

    def response(self, user_input, callback=None):
        
        token_counter = Counter()
        if isinstance(user_input, list):
            self.__transcript_history = user_input
            # 循环统计token数量，如果大于4096，则删除最早一条消息，直到token数量小于4096
            while token_counter.num_tokens_from_messages(self.__transcript_history) > 4096:
                if len(self.__transcript_history) == 1:
                    yield f"data: {json.dumps({'exceed': True})}\n\n"
                self.__transcript_history = self.__transcript_history[2:]
        else:
            # 统计token数量，如果大于4096，则返回exceed事件
            if token_counter.num_tokens_from_string(user_input) > 4096:
                yield f"data: {json.dumps({'exceed': True})}\n\n"
            self.__transcript_history += [{'role': 'user',
                                           'content': self.__user_input_prefix + user_input + self.__user_input_suffix}]

        if self.__is_azure_gpt4:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.__openai_api_key,
            }

            params = {
                'api-version': '2023-03-15-preview',
            }

            json_data = {
                'messages': self.__transcript_history,
                'max_tokens': self.__max_tokens,
                'temperature': self.__temperature,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                'stop': None,
            }

            url = 'https://azure.forkway.cn/openai/deployments/gpt-4/chat/completions'
            
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=json_data,
            )
            result = response.json()['choices'][0]['message']['content']
            yield result
        else:
            openai.api_key = self.__openai_api_key
            print('model:', self.__model)
            
            response = openai.ChatCompletion.create(
                model=self.__model,
                messages=self.__transcript_history,
                max_tokens=self.__max_tokens,
                temperature=self.__temperature,
                stream=self.__is_stream,
            )

            if self.__is_stream:
                self.__collected_messages = []
                self.__full_reply_content = ""
                for chunk in response:
                    chunk_message = chunk["choices"][0]["delta"]
                    self.__collected_messages.append(chunk_message)

                    if not chunk_message:
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        self.__full_reply_content = ''.join(
                            [m.get('content', '') for m in self.__collected_messages])
                        self.__transcript_history += [{'role': 'assistant',
                                                    'content': self.__full_reply_content}]
                        if self.__memory:
                            with self.__transcripts_db as db:
                                db.store_data(self.__table_name, self.__session_id,
                                            self.__column_name, self.__transcript_history)
                        if callback:
                            callback(self.__full_reply_content)
                        break
                    elif chunk_message.get('content'):
                        yield f"data: {json.dumps({'response': chunk_message['content']})}\n\n"

            else:
                golem_response = response['choices'][0]['message']['content']
                print('golem:', golem_response)
                self.__transcript_history += [{'role': 'assistant',
                                            'content': golem_response}]
                if self.__memory:
                    with self.__transcripts_db as db:
                        db.store_data(self.__table_name, self.__session_id,
                                    self.__column_name, self.__transcript_history)
                if callback:
                    callback(golem_response)
                yield golem_response
