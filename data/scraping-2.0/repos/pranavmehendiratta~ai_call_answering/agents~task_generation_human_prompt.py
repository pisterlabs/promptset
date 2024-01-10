from langchain.prompts import StringPromptTemplate

task_generation_human_prompt_v1 = """{input}"""

class TaskGenerationHumanPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        # Add all the partial variables for formatting
        kwargs.update(self.partial_variables)
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        return self.template.format(**kwargs)
    

task_generation_human_prompt = TaskGenerationHumanPromptTemplate(
    template=task_generation_human_prompt_v1,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input"]
)