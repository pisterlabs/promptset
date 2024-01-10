from typing import List, Union
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import re

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        llm_output = llm_output.replace('<|im_end|><|endoftext|>', '')
        llm_output = llm_output.split('Observation:')[0]
        llm_output = llm_output.split('Question:')[0]
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        else:
            # Parse out the action and action input
            regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                if llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.strip()},
                        log=llm_output,
                    )
                else:
                    return AgentFinish(
                        return_values={"output": None},
                        log=llm_output,
                    )
                raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            action = match.group(1).strip()
            action_input = match.group(2)
            # Return the action and action input
            return AgentAction(
                tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
            )