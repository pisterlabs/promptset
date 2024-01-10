import openai

from logic.handlers import build_StateChange, build_StateChangeCal
from logic.llm_driver import build_step
from logic.parsers import parse_action, parse_action_input, parse_after_prefix
from logic.prompts.main_prompt import main_prompt

import os
openai.api_key = os.getenv("OPENAI_API_KEY")


import xml.etree.ElementTree as ET
import re


def reply_once(chat_history, state, temperature):
    delta_message = ""
    observe_step = build_step(stop=['Observation:'], temperature=temperature)
    normal_step = build_step(temperature=temperature)

    handlers = {
        "StateChangeCal": build_StateChangeCal(chat_history, normal_step),
        "StateChange": build_StateChange(state),
    }

    message = observe_step(main_prompt(chat_history, delta_message, state))
    max_step = 6
    while message.find("Response:") == -1 and max_step > 0:
        if message.find("Action:") != -1:
            action = parse_action(message)
            action_input = parse_action_input(message)
            print("debug: action: "+action)
            print(f"debug: action_input: {action_input}")
            observation = handlers[action](action_input)
            delta_message = delta_message + message + "\nObservation:" + observation + "\n"
        message = observe_step(main_prompt(chat_history, delta_message, state))
        max_step -= 1
    print(message)
    return Response(parse_after_prefix(message, "Response:"))


class Response:
    def __init__(self, xml_data):
        # 解析XML数据
        xml_match = re.search(r'<response>.*?</response>', xml_data, re.DOTALL)
        if xml_match is None:
            print(xml_data)
            raise Exception("没有完整的xml")
        root = ET.fromstring(xml_data)
        self.line = root.find('line').text
        self.strategy = root.find('strategy').text

    def __str__(self):
        return f"Line: {self.line}, Strategy: {self.strategy}"




