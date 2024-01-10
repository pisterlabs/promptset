import os
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents import (
    AgentExecutor,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents import load_tools

chat_history = []

def get_agent(llm, *, chat_history: Optional[list] = None):
    chat_history = chat_history or []
    system_message = SystemMessage(
        content="あなたは計算を助けるアシスタントです"
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(
        memory_key="chat_history", llm=llm, max_token_limit=2000
    )

    for msg in chat_history:
        if "question" in msg:
            memory.chat_memory.add_user_message(str(msg.pop("question")))
        if "result" in msg:
            memory.chat_memory.add_ai_message(str(msg.pop("result")))

    tools = load_tools(["llm-math"], llm=llm)

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

    return agent_executor

def chat(question, chat_history):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    agent = get_agent(llm, chat_history=chat_history)
    return agent.invoke(
        #{"input": question, "chat_history": chat_history}
        {"input": question}
        )


#res = chat("1+1は？", [])
# res = chat("さらに1足すと？", [{"question": "1+1は？"}, {"result": "2です"}])
# print("\n\n",res)


# 会話ループ
user = ""
while user != "入力してください:exit":
    user = input("入力してください:")
    user_input = user[len("入力してください:"):]
    
    print(f"user_input: {user_input}")
    ai_output = chat(user_input, chat_history)

    new_history = []
    new_history.append(
        dict(question=user_input)
    )
    new_history.append(
        dict(result=ai_output)
    )
    print("")
    print(ai_output)
