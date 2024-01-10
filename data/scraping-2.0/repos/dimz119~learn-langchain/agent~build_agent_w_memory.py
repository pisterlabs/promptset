from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(
                    functions=[
                        format_tool_to_openai_function(t) for t in tools # type: ignore
                    ]) 


from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

chat_history = []

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # type: ignore

input1 = "how many letters in the word educa?"
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})
print(result)
"""
{'input': 'how many letters in the word educa?', 'chat_history': [], 'output': 'There are 5 letters in the word "educa".'}
"""

chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)
print(agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history}))
"""
{
    'input': 'is that a real word?',
    'chat_history': [
        HumanMessage(content='how many letters in the word educa?'),
        AIMessage(content='There are 5 letters in the word "educa".')], 
    'output': 'No, "educa" is not a real word in English.'}
"""
