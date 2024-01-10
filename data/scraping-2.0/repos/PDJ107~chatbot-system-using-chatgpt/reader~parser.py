from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from messages import SpecificFCM
from pydantic import Extra
from typing import Union
import re


class CustomOutputParser(AgentOutputParser, extra=Extra.allow):

    def __init__(self, tool_names: list, message_client: SpecificFCM = None):
        super().__init__()
        self.message_client = message_client
        self.tool_names = tool_names

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"

        # print(llm_output)

        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        if action in self.tool_names and self.message_client is not None:
            self.message_client.send_message(
                '\n'.join(["Action: "+action, "Action Input: "+action_input]),
                debug=True
            )

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
