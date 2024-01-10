#%%
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough

memory = ConversationBufferMemory(
    return_messages=True
)

def load_memory():
    x = memory.load_memory_variables({})
    return {"history": x["history"]}

# %%

memory.save_context({"input": "영감"} , {"output": "왜불러"})
memory.save_context({"input": "뒤뜰어 매어놓은 강아지 한마리 보았소?"} , {"output": "보았지"})
memory.save_context({"input": "어쨋수"} , {"output": "먹엇지"})
memory.save_context({"input": "잘했군!"} , {"output": "자~알 했어?"})

_memory = load_memory()
print(_memory)
#%%
memory.save_context({"input": "잘했군 잘했군 잘했어"} , {"output": "........"})

_memory = load_memory()
print(f"history length: {len(_memory['history'])} ")
print(_memory)
# %%
