from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI



@cl.on_chat_start
def start():
    prompt = ChatPromptTemplate.from_messages(
        [

            MessagesPlaceholder(
                variable_name="history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "You are an enthusiastic computer software and hardware expert that is here to help people learn about somputers in general. Assume that questions asked are from people who have little knowledge of computers. Provide snippets and learning resources for each of your answers. Be enthusiatic in your responses and DON'T FORGET TO PROVIDE EXAMPLES!!!!. {input}"
            ),  # Where the human input will injected
        ]
    )

    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    # llm = ChatOpenAI(streaming=True, model="gpt-3.5-turbo", temperature=1)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1)
    llm_chain = ConversationChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    cl.user_session.set("agent_chain", llm_chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("agent_chain")  # type: LLMChain

    res = await chain.arun(
        input=message.content, callbacks=[cl.LangchainCallbackHandler()]
    )

    print(f"chain: {chain}")

    await cl.Message(content=res).send()