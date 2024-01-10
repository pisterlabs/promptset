import json
import os

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from marvin_hue.basics import LightConfig
from marvin_hue.controllers import HueController
from services.parsers import LightConfigJsonParser
from services.prompts import PromptInstructionsHue, PromptResponseHue

load_dotenv(find_dotenv())


class HueChat:
    _hue = HueController(ip_address=os.environ["bridge_ip"])

    def __init__(self):
        self._prompt_instructions = ChatPromptTemplate.from_template(PromptInstructionsHue.get())
        self._prompt_result = ChatPromptTemplate.from_template(PromptResponseHue.get())
        self._response = ""
        self._instructions = ""
        self._json = {"result": ""}

    @property
    def response(self):
        return self._response

    @staticmethod
    def _parser_json(data: str) -> dict:
        try:
            return LightConfigJsonParser(data).output
        except Exception:
            raise

    def send_instructions(self, instruction: str, llm="gpt-3.5-turbo"):
        # Solicita à LLM para criar a configuração de cores.
        self._instructions = instruction
        messages = self._prompt_instructions.format_messages(text=instruction)
        chat = ChatOpenAI(temperature=0.0, model=llm)
        response = chat(messages)
        # Recupera o JSon de resposta.
        self._json = self._parser_json(response.content)
        return self

    def apply(self):
        # Aplica a configuração nas luzes do ambiente.
        try:
            lc = LightConfig().from_dict(self._json)
            self._hue.apply_light_config(light_config=lc)
        except Exception as e:
            self._response = f"""
            ERRO: {str(e)}
            
            {json.dumps(self._json, indent=2, ensure_ascii=False)}
            """
        return self

    def get_response(self, llm="gpt-3.5-turbo"):
        msgs = self._prompt_result.format_messages(
            instructions=self._instructions,
            text=json.dumps(self._json)
        )
        chat = ChatOpenAI(temperature=0.0, model=llm)
        res = chat(msgs)
        self._response = res.content
        return self


if __name__ == '__main__':
    inst = """
        Crie uma configuração de cores para um dia de sol de primavera repleto de flores coloridas.
        """
    _llm = "gpt-3.5-turbo"
    _chat = HueChat().send_instructions(instruction=inst, llm=_llm).apply().get_response(_llm)
    print(_chat.response)
