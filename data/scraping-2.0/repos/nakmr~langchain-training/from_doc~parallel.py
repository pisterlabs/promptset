from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# model = ChatOpenAI()
# joke_chain = ChatPromptTemplate.from_template("telll me a joke about {topic}") | model | StrOutputParser()
# poem_chain = ChatPromptTemplate.from_template("write a poem about {topic}") | model | StrOutputParser()

# map_chain = RunnableParallel(
#     joke=joke_chain,
#     poem=poem_chain
# )

# result = map_chain.invoke({"topic": "accenture"})
# print(result)

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

print(runnable.invoke({"num": 5}))