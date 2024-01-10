import json
import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from .prompts import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_ACTION = "```Respuesta final```:"
TOOL_NAMES = ['La ley', 'La Ley']
WEAK_FINAL_ANSWER_ACTION = "Respuesta final:"


class CustomChatOutputParser(AgentOutputParser):
    # conversacion no optimizada para chat con historico
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            action = text.split("```")[1]
            response = json.loads(action.strip())
            if response['action'] not in TOOL_NAMES:
                raise ValueError(f"Action {response['action']} not in {TOOL_NAMES}")
            return AgentAction(response["action"], response["action_input"], text)
        except Exception:
            if FINAL_ANSWER_ACTION in text:
                return AgentFinish(
                    {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
                )
            elif WEAK_FINAL_ANSWER_ACTION in text:
                return AgentFinish(
                    {"output": text.split(WEAK_FINAL_ANSWER_ACTION)[-1].strip()}, text
                )
            raise OutputParserException(f"Could not parse LLM output: {text}")


# from .prompts_eval import EVAL_FORMAT_INSTRUCTIONS

# class EvalConvoOutputParser(AgentOutputParser):
#     # conversacion optimizada para chat con historico
#     ai_prefix: str = "AI"

#     def get_format_instructions(self) -> str:
#         return EVAL_FORMAT_INSTRUCTIONS

#     def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
#         if f"{self.ai_prefix}:" in text:
#             return AgentFinish(
#                 {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
#             )
#         regex = r"Acción: (.*?)[\n]*Entrada a la acción: (.*)"
#         match = re.search(regex, text)
#         if not match:
#             raise OutputParserException(f"Could not parse LLM output: `{text}`")
#         action = match.group(1)
#         action_input = match.group(2)
#         return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)


# from .prompts_magistrado import MAGISTRADO_FORMAT_INSTRUCTIONS

# class MagistradoConvoOutputParser(AgentOutputParser):
#     # conversacion optimizada para chat con historico
#     ai_prefix: str = "AI"

#     def get_format_instructions(self) -> str:
#         return MAGISTRADO_FORMAT_INSTRUCTIONS

#     def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
#         if f"{self.ai_prefix}:" in text:
#             return AgentFinish(
#                 {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
#             )
#         regex = r"Acción: (.*?)[\n]*Entrada a la acción: (.*)"
#         match = re.search(regex, text)
#         if not match:
#             raise OutputParserException(f"Could not parse LLM output: `{text}`")
#         action = match.group(1)
#         action_input = match.group(2)
#         return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)


# from .prompts_sragent import AGENT_FORMAT_INSTRUCTIONS

# class SrAgentConvoOutputParser(AgentOutputParser):
#     # conversacion optimizada para chat con historico
#     ai_prefix: str = "AI"

#     def get_format_instructions(self) -> str:
#         return AGENT_FORMAT_INSTRUCTIONS

#     def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
#         if f"{self.ai_prefix}:" in text:
#             return AgentFinish(
#                 {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
#             )
#         regex = r"Acción: (.*?)[\n]*Entrada a la acción: (.*)"
#         match = re.search(regex, text)
#         if not match:
#             raise OutputParserException(f"Could not parse LLM output: `{text}`")
#         action = match.group(1)
#         action_input = match.group(2)
#         return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)
