from langchain.agents import AgentExecutor, XMLAgent, tool
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"

tool_list = [search]

# Get prompt to use
prompt = XMLAgent.get_default_prompt()
"""
input_variables=['intermediate_steps', 'question', 'tools'] 
messages=[
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['question', 'tools'], 
            template="You are a helpful assistant. Help the user answer any questions.\n\n
            You have access to the following tools:\n\n{tools}\n\n
            In order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. 
            You will then get back a response in the form <observation></observation>\n
            For example, if you have a tool called 'search' that could run a google search, 
            in order to search for the weather in SF 
            you would respond:\n\n
            <tool>search</tool>
            <tool_input>weather in SF</tool_input>\n
            <observation>64 degrees</observation>\n\n
            When you are done, respond with a final answer between 
            <final_answer></final_answer>. 
            
            For example:\n\n<final_answer>The weather in SF is 64 degrees</final_answer>\n\n
            Begin!\n\nQuestion: {question}")), 
    AIMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=['intermediate_steps'],
            template='{intermediate_steps}'))]
"""
# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

agent = (
    {
        "question": lambda x: x["question"],
        "intermediate_steps": lambda x: convert_intermediate_steps(
            x["intermediate_steps"]
        ),
    }
    | prompt.partial(tools=convert_tools(tool_list)) # prompt.partial: Get a new ChatPromptTemplate with some input variables already filled in
    | llm.bind(stop=["</tool_input>", "</final_answer>"])
    | XMLAgent.get_default_output_parser()
)
"""
Expects output to be in one of two formats.

If the output signals that an action should be taken,
should be in the below format. This will result in an AgentAction
being returned.

```
<tool>search</tool>
<tool_input>what is 2 + 2</tool_input>
```

If the output signals that a final answer should be given,
should be in the below format. This will result in an AgentFinish
being returned.

```
<final_answer>Foo</final_answer>
```
"""

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True) # type: ignore

print(agent_executor.invoke({"question": "whats the weather in New york?"}))
