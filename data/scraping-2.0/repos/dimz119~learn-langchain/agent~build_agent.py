from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print(llm.invoke("how many letters in the word educa?"))
"""
content='There are 6 letters in the word "educa".'
"""


# Build a custom tool
from langchain.agents import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]




from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# input: should be a string containing the user objective
# agent_scratchpad: should be a sequence of messages that contains 
#   the previous agent tool invocations and 
#   the corresponding tool outputs


# How does the agent know what tools it can use? 
# In this case we’re relying on OpenAI function calling LLMs, 
#   which take functions as a separate argument and 
#   have been specifically trained to know when to invoke those functions.
from langchain.tools.render import format_tool_to_openai_function

llm_with_tools = llm.bind(
                    functions=[
                        format_tool_to_openai_function(t) for t in tools # type: ignore
                    ]) 


from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

print(agent.invoke({"input": "how many letters in the word educa?", "intermediate_steps": []}))
"""
tool='get_word_length' 
tool_input={'word': 'educa'} 
log="\nInvoking: `get_word_length` with `{'word': 'educa'}`\n\n\n" 
message_log=[
    AIMessage(
        content='',
        additional_kwargs={
            'function_call': {
                'arguments': '{\n  "word": "educa"\n}',
                'name': 'get_word_length'}})]
"""



# Define Runtime
from langchain.schema import AgentFinish

user_input = "how many letters in the word educa?"
intermediate_steps = []
while True:
    output = agent.invoke(
        {
            "input": user_input,
            "intermediate_steps": intermediate_steps,
        }
    )

    # AgentFinish: This signifies that the agent has finished and 
    #               should return to the user
    if isinstance(output, AgentFinish):
        final_result = output.return_values["output"]
        break
    else:
        # AgentAction: This represents the action an agent should take
        print(f"TOOL NAME: {output.tool}")
        print(f"TOOL INPUT: {output.tool_input}")
        tool = {"get_word_length": get_word_length}[output.tool]
        """
        tool = {"x": lambda a: a+1}['x']
        result = tool(3)  # This will set 'result' to 4
        """
        observation = tool.run(output.tool_input)
        intermediate_steps.append((output, observation))
print(final_result)
"""
TOOL NAME: get_word_length
TOOL INPUT: {'word': 'educa'}
There are 5 letters in the word "educa".
"""



# Using AgentExcutor
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # type: ignore
print(agent_executor.invoke({"input": "how many letters in the word educa?"}))
"""
> Entering new AgentExecutor chain...

Invoking: `get_word_length` with `{'word': 'educa'}`


There are 5 letters in the word "educa".

> Finished chain.
{'input': 'how many letters in the word educa?', 'output': 'There are 5 letters in the word "educa".'}
"""