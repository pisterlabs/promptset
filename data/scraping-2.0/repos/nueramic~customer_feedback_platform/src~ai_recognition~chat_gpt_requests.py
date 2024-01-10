import json
from sqlalchemy.engine import Engine
import pandas as pd
import os
import openai
from datetime import datetime
import toml


class AnalyzeFeedback:

    def __init__(self, pg_conn: Engine, openai_api_key: str):
        openai.api_key = openai_api_key
        self.pg_conn = pg_conn
        self.id_feedback = ''
        self.chat_response = ''
        self.gpt_config = toml.load('gpt_message_config.toml')

    def analyze_feedback(self,
                         id_feedback: str,
                         text_feedback: str,
                         rating: str = 1,
                         max_rating: str = 1
                         ):
        """

        :param id_feedback:
        :param text_feedback:
        :param rating:
        :param max_rating:
        :return:
        """

        message = f"""
        {self.gpt_config['gpt_role']['instruction']}
        отзыв: \n {text_feedback[:5000]} 
        оценка: {rating} / {max_rating}
        """

        messages = [{'role': 'assistant', 'content': message}]

        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.2
        )

        self.id_feedback: str = id_feedback
        self.chat_response: str = chat.choices[0].message.content

    def save_to_table(self):
        """
        insert into table gpt response
        """

        try:
            resp_json = json.dumps(json.loads(self.chat_response), ensure_ascii=False)
            df = pd.DataFrame(
                {
                    'id_feedback': [self.id_feedback],
                    'json_gpt_resp_content': [resp_json],
                    'dtime_updated': [datetime.now()]
                })
            df.to_sql('ai_responses', self.pg_conn, schema='prod', index=False, if_exists='append')

            return resp_json

        except Exception as e:
            print(e)
