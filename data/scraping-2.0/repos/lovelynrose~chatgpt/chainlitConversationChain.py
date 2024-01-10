# Set the config field hide_cot to true in the .chainlit/config.toml file. Then restart the chainlit server.
# Change chainlit.md

import chainlit as cl

from langchain import PromptTemplate, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import openai

openai.api_key = yourkey
llm_openai=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613",openai_api_key = openai.api_key)

# To use local llama llm model
# Downlaod from : https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF/tree/main
'''
from langchain.llms import LlamaCpp
llm = LlamaCpp(
    model_path="d:\\python code\\models\\airoboros-l2-7b-2.1.Q4_K_M.gguf",
)
'''

template = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""

@cl.on_chat_start
def start():
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    llm_chain = ConversationChain(
        prompt=prompt,
        llm=llm_openai,
        verbose=True,
        memory=ConversationBufferMemory(),
    )
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print("Result : ",res)
    # "res" is a Dict. For ConversationChain, we get the response by reading the "response" key.
    # "input" and "history" are the other keys

    await cl.Message(content=res["response"]).send()

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"ConversationChain": "Agent", "Chatbot": "Assistant"}
    return rename_dict.get(orig_author, orig_author)

