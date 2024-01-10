from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

model1 = ChatOpenAI(model="gpt-3.5-turbo")
model2 = ChatOpenAI(model="gpt-4")

prompt1 = ChatPromptTemplate.from_template("""
Apenas traduza o texto a seguir:
{text}
"""
)

prompt2 = ChatPromptTemplate.from_template( """
You are now my personal travel agent. Act as someone who has immense travel
experience and knows the best places in the world to do certain activities. I
want to know where I should go to {activity}. Give the answers as a list of
items, no bigger than 5 items. Only respond with the list of 5 items, no
summarizing statement, forewords or warnings or even explanations are required.
"""
)

chain1 = {"text": RunnablePassthrough()} | prompt1 | model1

chain2 = {"activity": chain1} | prompt2 | model2

chain3 = {"text": chain2} | prompt1 | model1

for s in chain3.stream("caminhar na praia"):
    print(s.content, end="", flush=True)
print()
