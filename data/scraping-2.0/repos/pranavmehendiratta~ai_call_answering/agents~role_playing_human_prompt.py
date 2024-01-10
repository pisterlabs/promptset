from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool
from typing import List

human_template_v4 = """executed_function_history:
{function_memory}

conversation_history:
{history}

[start]
question: {input}
{agent_scratchpad}"""

class RolePlayingHumanMessagePromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    
    # Memory hack to save function execution to prevent re-execution
    long_term_function_memory = ""
    current_function_memory = ""

    def format(self, **kwargs) -> str:
        # Add all the partial variables for formatting
        kwargs.update(self.partial_variables)
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")

        if len(intermediate_steps) == 0:
            self.long_term_function_memory = self.long_term_function_memory + self.current_function_memory

        kwargs["function_memory"] = self.long_term_function_memory
        #print("CustomHumanMessagePromptTemplate ----->")
        #print(intermediate_steps)
        #print(" <-------------------------")

        thoughts = ""
        self.current_function_memory = ""
        for agent_action, function_result  in intermediate_steps:
            thoughts += agent_action.log
            thoughts += f"\nfunction_return: {function_result}\nfunction_return_extraction:"
            self.current_function_memory = self.current_function_memory + f"{agent_action.tool}({agent_action.tool_input}) -> {function_result}\n"

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        #print("CustomHumanMessagePromptTemplate ----->")
        #print(intermediate_steps)
        #print(" <------------------------- Long Term Function Memory ----->")
        #print(self.long_term_function_memory)
        #print(" <------------------------- Current Function Memory ----->")
        #print(self.current_function_memory)
        #print(" <-------------------------")

        #conversation_history += "\nFunction_result: {function_result}\nEvaluation:"

        #print("CustomHumanMessagePromptTemplate agent_scratchpad ----->")
        #print(kwargs["agent_scratchpad"])
        #print(" <-------------------------")

        return self.template.format(**kwargs)
    


role_playing_human_prompt = RolePlayingHumanMessagePromptTemplate(
    template=human_template_v4,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)