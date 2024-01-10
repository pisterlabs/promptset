import os
import sys
import constants
import openai

from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.schema.messages import HumanMessage, AIMessage

os.environ["OPENAI_API_KEY"] = constants.APIKEY

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

@tool
def get_sum(a: int, b: int) -> int:
    """Returns the sum of two numbers."""
    return a + b

@tool
def get_difference(a: int, b: int) -> int:
    """Returns the difference of two numbers."""
    return a - b

tools = [get_word_length,
         get_sum,
         get_difference]

MEMORY_KEY = "chat_history"

# Prompt that will be used to chat with the bot.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Hola, soy un bot que responde diferencia entre operaciones matemáticas y longitud de palabras. ¿Qué quieres hacer?"),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# LLM that will be used to generate the responses.
llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps']),
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chat_history = []

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    result = agent_executor.invoke({"input": query, "chat_history": chat_history})

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result['output']))
    query = None
