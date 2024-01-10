from operator import itemgetter
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

runnable = RunnableParallel(
    passed=RunnablePassthrough()
)

print(runnable.invoke({"num": 1}))
"""
{'passed': {'num': 1}}
"""

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)

print(runnable.invoke({"num": 1}))
"""
{'passed': {'num': 1}, 'extra': {'num': 1, 'mult': 3}, 'modified': 2}
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel

model = ChatOpenAI()

joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
)

# easy to execute multiple Runnables in parallel
map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

print(map_chain.invoke({"topic": "bear"}))
"""
{
    'joke': AIMessage(content="Why don't bears wear shoes?\n\nBecause they have bear feet!"), 
    'poem': AIMessage(content="In the wild's embrace, a bear roams free,\nMajestic strength, nature's symphony.")
}
"""
