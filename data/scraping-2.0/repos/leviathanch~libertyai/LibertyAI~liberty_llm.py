import time

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

import requests
from LibertyAI.liberty_config import get_configuration

class LibertyLLM(LLM):

    endpoint: str
    #temperature: float
    #max_tokens: int
    echo: bool =  False

    @property
    def _llm_type(self) -> str:
        return "liberty"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        uuid = self.submit_partial(prompt, stop)
        if not uuid:
            return "[DONE]"

        ret = ""
        text = ""
        i = 0
        while text != "[DONE]":
            text = self.get_partial(uuid, i)
            i += 1
            if text != "[DONE]":
                ret += text

        return ret

    def submit_partial(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = prompt.replace("[DONE]", b'\xf0\x9f\x96\x95'.decode()).replace("[BUSY]", b'\xf0\x9f\x96\x95'.decode())
        config = get_configuration()
        jd = {'text' : prompt,}
        if stop:
            jd['stop'] = stop

        try:
            response = requests.post(
                self.endpoint+'/submit',
                json = jd,
            )
            reply = response.json()
        except:
            return None

        if 'uuid' in reply:
            return reply['uuid']
        else:
            return None

    def get_partial(self, uuid, index):
        text = "[DONE]"
        config = get_configuration()
        jsd = {'uuid' : uuid, 'index': str(index) }
        try:
            response = requests.post(
                self.endpoint+'/fetch',
                json = jsd,
            )
            reply = response.json()
        except:
            return "[DONE]"

        if 'text' in reply:
            text = reply['text']

        return text
