from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool
from typing import List

test_human_human_prompt_v1 = """Me: Hello, this is John at Timeplated Restaurant. How may I help you?
Your Response: {agent_scratchpad}"""

class TestHumanHumanMessagePromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        # Add all the partial variables for formatting
        kwargs.update(self.partial_variables)
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")

        thoughts = ""
        for agent_action, assistant_response  in intermediate_steps:
            thoughts += agent_action.log
            thoughts += f"\nMe: {assistant_response}\nYour Response:"

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        return self.template.format(**kwargs)
    

test_human_human_prompt = TestHumanHumanMessagePromptTemplate(
    template=test_human_human_prompt_v1,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["intermediate_steps"]
)