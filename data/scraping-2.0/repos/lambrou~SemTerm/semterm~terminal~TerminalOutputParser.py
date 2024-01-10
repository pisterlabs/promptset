import json
from abc import ABC
from typing import Union
from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.schema import AgentAction, AgentFinish
from semterm.agent.TerminalAgentPrompt import FORMAT_INSTRUCTIONS
from semterm.langchain_extensions.schema import AgentMistake


class TerminalOutputParser(ConvoOutputParser, ABC):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        text = text.strip().replace("\xa0", " ")
        start_positions = [i for i, c in enumerate(text) if c == "{"]
        end_positions = [i for i, c in enumerate(text) if c == "}"]

        for start in start_positions:
            for end in end_positions:
                if start < end:  # ensure the end position is after the start
                    try:
                        cleaned_output = text[start : end + 1]
                        response = json.loads(cleaned_output)
                        action, action_input = (
                            response["action"],
                            response["action_input"],
                        )
                        if action == "Final Answer":
                            return AgentFinish({"output": action_input}, text)
                        else:
                            return AgentAction(action, action_input, text)
                    except json.JSONDecodeError:
                        return AgentMistake(text, text)

        # If we reach this point, no valid JSON was found in the text
        return AgentFinish({"output": text}, text)
