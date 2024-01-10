from langchain.prompts import StringPromptTemplate
from tools import ALL_TOOLS


# Set up the base template
template = """You will be passed a JSON regarding an invoice from a company, your goal is to
write a brief Description of the services provided by the company.
Keep the description to one short sentence. You can use the following tools
in order to search for more information about the company:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to keep the Description to a single short sentence, as if writing a new invoice.

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps", [])  # Added a default empty list
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Since tools are static, just use ALL_TOOLS
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in ALL_TOOLS]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in ALL_TOOLS])

        return self.template.format(**kwargs)
    
prompt = CustomPromptTemplate(
    template=template,  # 'template' is the string defined in 'template.py'
    input_variables=["input", "intermediate_steps"],
)