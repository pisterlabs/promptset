import re
from typing import Tuple
from pkg_resources import resource_filename
import json
from bots.bot import Bot
from bots.gpt.openai_util import openai_call


class GptBotNavigator(Bot):
    def __init__(self, map_id, cs_strategy):
        super().__init__(cs_strategy)

        kb_path = resource_filename('bots', f'gpt/nav_kw.json')
        with open(kb_path, 'r') as f:
            kb = json.load(f)

        map_kw = kb[f'{map_id}']
        map_suffix_kw = kb[f'{map_id}_suffix']

        system_prefix1 = kb['system_common_prefix1']
        system_prefix2 = kb['system_common_prefix2']
        system_prefix3 = kb['system_common_prefix3']
        system_suffix = kb['system_common_suffix1']
        system_map_prefix = kb['system_common_map_prefix']

        self.final_object = kb[f'{map_id}_final_object']

        system_content = f'{system_prefix1}\n{system_prefix2}\n{system_prefix3}\n{system_map_prefix}\n{map_kw}\n{map_suffix_kw}\n{system_suffix}'
        if self.is_spanish_cs_strategy:
            system_content = f'{system_content}\nPlease respond in Spanish only, The user can chat in either English or Spanish.'
        self.messages = [
            {"role": "system", "content": system_content},
            {'role': 'assistant', 'content': self.welcome_str}
        ]

    def __is_finished(self, bot_resp: str):
        t = bot_resp.lower()
        if "*" in t:
            if 'finished' in t: return True
            if 'misión cumplida!' in t: return True
            if 'terminado' in t: return True
            if 'finalizado' in t: return True
            if 'completado' in t: return True
            if 'logrado' in t: return True
            if 'felicidades' in t: return True
            if 'finalizado'in t: return True

        match = bool(re.match(f"(.*)((reached|arrive|arrived) (.*) {self.final_object})(.*)", t))
        match |= bool(re.match(f"(.*)((reached|arrive|arrived) (.*) final (destination|object))(.*)", t))
        match |= bool(re.match("(.*)((alcanzado|llegar|llegado) (.*) destino (final|objeto))(.*)", t))
        match |= bool(re.match("(.*)((alcanzado|llegar|llegado) (.*) final (destino|objeto))(.*)", t))

        return match


    def call(self, user_msg, user_state=None) -> Tuple[list[str], bool]:
        self.messages.append({'role': 'user', 'content': user_msg})
        msg, resp = openai_call(self.messages)
        is_finished = self.__is_finished(msg)
        if not resp:
            return [msg], is_finished

        post_proc_msgs = self.informal_post_process(msg)
        for pp_msg in post_proc_msgs:
            self.messages.append({'role': 'assistant', 'content': pp_msg})
        return post_proc_msgs, is_finished


    def db_push(self) -> dict:
        return {}

    def db_load(self, data):
        pass