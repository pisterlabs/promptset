import json
from typing import Union, Callable

from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException

from langchain.input import get_colored_text

FINAL_ANSWER_ACTION = "Final Answer:"


class ExtendedChatOutputParser(ChatOutputParser):

    action_detector_func: Callable

    def __init__(self, action_detector_func: Callable):
        super().__init__(action_detector_func=action_detector_func)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text

        try:
            action = self.action_detector_func(text)
            response = json.loads(action.strip())
            includes_action = "action" in response
            if includes_answer and includes_action:
                raise OutputParserException(
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
            print(get_colored_text(f"Tool: {response['action']}", "blue"))
            print(get_colored_text(f"Input: {response['action_input']}", "blue"))
            print()
            return AgentAction(
                response["action"], response.get("action_input", {}), text
            )

        except Exception as e:
            if not includes_answer:
                raise OutputParserException(f"Could not parse LLM output: {text}: {str(e)}")
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )