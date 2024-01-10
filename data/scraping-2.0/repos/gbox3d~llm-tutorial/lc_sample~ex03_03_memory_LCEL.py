#%%
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StreamingStdOutCallbackHandler


import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')
llm = ChatOpenAI(temperature=0.1,
                 streaming=True,
                 callbacks=[
                     StreamingStdOutCallbackHandler(),
                 ]
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    return_messages=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 사람들과 도움이되는 대화를 하는 챗봇입니다. 대화는 한글로 진행됩니다."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def load_memory(_):
    return memory.load_memory_variables({})["history"]


chain = RunnablePassthrough.assign(history=load_memory) | prompt | llm


def invoke_chain(question):
    result = chain.invoke({"question": question})
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print('\n\n')
    print(result)
    
#%%
result = invoke_chain("여기 딸기가 3개있습니다.")        
# %%
result = invoke_chain("하나를 먹었습니다.")

# %%
invoke_chain("지금 딸기가 몇개 남았습니까?")
# %%
load_memory({})
