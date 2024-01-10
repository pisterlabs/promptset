from langserve import RemoteRunnable
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

category_chain = RemoteRunnable("http://localhost:8000/category_chain/")
joke_chain = RemoteRunnable("http://localhost:8000/joke/")
openai_chain = RemoteRunnable("http://localhost:8000/openai/")


print(category_chain.invoke({"text": "colors"}))
# >> ['red', 'blue', 'green', 'yellow', 'orange']

print(joke_chain.invoke({'topic':"lion"}))

prompt = [
    SystemMessage(content='Act like either a cat or a parrot.'),
    HumanMessage(content='Hello!')
]

# Supports astream
for msg in anthropic.astream(prompt):
    print(msg, end="", flush=True)

prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)

# Can define custom chains
chain = prompt | RunnableMap({
    "openai": openai,
    # "anthropic": anthropic,
})

print(chain.batch([{ "topic": "parrots" }, { "topic": "cats" }]))
