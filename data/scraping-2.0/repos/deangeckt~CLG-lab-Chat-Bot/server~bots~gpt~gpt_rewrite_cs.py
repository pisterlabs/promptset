from typing import List
from bots.gpt.openai_util import openai_call
from bots.cs_unit import CSUnit


class GPTCodeSwitchReWrite(CSUnit):
    """
    This unit uses prompt to force GPT to re-write the bot response in a mixed language
    -----UNUSED-----
    """

    def __init__(self):
        super().__init__()
        self.chat = []
        self.system_content = "You are an English-Spanish bilingual speaker. Your responses MUST include words from both languages, and NOT a translation."
        self.user_prefix = "Rewrite the following sentence in mixed English-Spanish:"

    def call(self, user_msg: str, en_bot_resp: List[str]) -> List[str]:
        resp = []
        for bot_prev_msg in en_bot_resp:
            user_content = f'{self.user_prefix} "{bot_prev_msg}"'
            messages = [
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": user_content}
            ]
            msg, _ = openai_call(messages)
            resp.append(msg)
        return resp

    def db_push(self) -> dict:
        return {}

    def db_load(self, data):
        pass

    def is_switched(self):
        return True
