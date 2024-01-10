import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish


# Reference: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_tool-using_agent_with_Langchain.ipynb
class CustomOutputParser(AgentOutputParser):
    ai_prefix: str = "BOT"
    verbose: bool = False

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if f"{self.ai_prefix}" in llm_output:
            return AgentFinish(
                return_values={
                    "output": llm_output.split(f"{self.ai_prefix}:")[-1].strip()
                },
                log=llm_output,
            )

        # parse out the action and action input
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            # TODO: figure a more natural way to deal with parsing error
            # return AgentFinish(
            #     return_values={
            #         "output": "Sorry, I don't really have a good answer to your query at the moment. Perharps, let me check, and get back to you later?"
            #     },
            #     log=llm_output,
            # )
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )
