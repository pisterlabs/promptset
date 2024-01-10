import re
from typing import Tuple
from pkg_resources import resource_filename
import json
from bots.bot import Bot
from bots.gpt.openai_util import openai_call


class GptBotInstructor(Bot):
    def __init__(self, map_id, cs_strategy):
        super().__init__(cs_strategy)

        kb_path = resource_filename('bots', f'gpt/ins_kw.json')
        with open(kb_path, 'r') as f:
            kb = json.load(f)

        map_kw = kb[f'{map_id}']

        system_prefix = kb['system_common_prefix']
        system_suffix = kb['system_common_suffix']

        system_map_prefix = kb['system_common_map_prefix']

        system_content = f'{system_prefix}\n{system_map_prefix}\n{map_kw}\n{system_suffix}'
        if self.is_spanish_cs_strategy:
            system_content = f'{system_content}\nPlease respond in Spanish only, The user can chat in either English or Spanish.'
        self.messages = [
            {"role": "system", "content": system_content},
            {'role': 'assistant', 'content': self.welcome_str}
        ]

    def call(self, user_msg, user_state=None) -> Tuple[list[str], bool]:
        self.messages.append({'role': 'user', 'content': user_msg})
        msg, resp = openai_call(self.messages)
        if not resp:
            return [msg], False

        post_proc_msgs = self.informal_post_process(msg)
        for pp_msg in post_proc_msgs:
            self.messages.append({'role': 'assistant', 'content': pp_msg})
        return post_proc_msgs, False


    def db_push(self) -> dict:
        # chat is too long for DB
        # return {'chat': self.messages}
        return {}

    def db_load(self, data):
        pass
        # self.messages = []
        # for chat_ele in data['chat']:
        #     self.messages.append({'role': chat_ele['role'], 'content': chat_ele['content']})
