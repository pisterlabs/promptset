from langchain.agents import initialize_agent
from langchain.llms import GPT4All
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# llm = OpenAI(openai_api_key="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")
llm = GPT4All()

chat = ChatOpenAI(temperature=0)
chat.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# >> AIMessage(content="J'aime programmer.", additional_kwargs={})
#

# from platform import python_version
# print(python_version())

initialize_agent()